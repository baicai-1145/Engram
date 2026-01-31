from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


def expand_data_files(paths: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for p in paths:
        # Expand globs like OpenThoughts3-1.2M/data/train-*.parquet
        matches = glob.glob(p)
        if matches:
            out.extend(sorted(matches))
        else:
            out.append(p)
    # De-dup while preserving order.
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return tuple(uniq)

def _normalize_role(role: str) -> str:
    r = str(role).strip().lower()
    if r in ("human", "user"):
        return "user"
    if r in ("gpt", "assistant"):
        return "assistant"
    if r in ("system",):
        return "system"
    return r or "unknown"


def render_conversations_simple(
    conversations: Any,
    *,
    from_field: str = "from",
    value_field: str = "value",
    add_generation_prompt: bool = False,
) -> str:
    """Render OpenThoughts-style conversations into a plain-text transcript.

    This avoids relying on a tokenizer chat template (KISS) and works for CPT.
    """
    if not isinstance(conversations, list):
        raise ValueError(f"conversations must be a list, got {type(conversations)}")

    parts: List[str] = []
    for t in conversations:
        if not isinstance(t, dict):
            continue
        role = _normalize_role(t.get(from_field, ""))
        value = t.get(value_field, "")
        if not isinstance(value, str):
            continue
        prefix = {"system": "System", "user": "User", "assistant": "Assistant"}.get(role, role)
        parts.append(f"{prefix}: {value.strip()}")

    text = "\n\n".join(parts).strip()
    if add_generation_prompt:
        text = (text + "\n\nAssistant:").strip()
    return text


def extract_prompt_response_from_conversations(
    conversations: Any,
    *,
    from_field: str = "from",
    value_field: str = "value",
) -> Tuple[str, str]:
    """Convert a conversation list into a single (prompt, response) pair for SFT.

    Strategy:
    - find the last assistant message; use it as response
    - prompt is all turns before it, rendered with simple role prefixes, and ends with 'Assistant:'.
    """
    if not isinstance(conversations, list):
        raise ValueError(f"conversations must be a list, got {type(conversations)}")

    last_asst_idx = None
    for i in range(len(conversations) - 1, -1, -1):
        t = conversations[i]
        if not isinstance(t, dict):
            continue
        role = _normalize_role(t.get(from_field, ""))
        if role == "assistant":
            last_asst_idx = i
            break
    if last_asst_idx is None:
        raise ValueError("No assistant turn found in conversations.")

    resp_turn = conversations[last_asst_idx]
    response = resp_turn.get(value_field, "")
    if not isinstance(response, str):
        raise ValueError(f"Assistant '{value_field}' must be a string, got {type(response)}")

    prompt_turns = conversations[:last_asst_idx]
    prompt = render_conversations_simple(
        prompt_turns, from_field=from_field, value_field=value_field, add_generation_prompt=True
    )
    return prompt, response.strip()


def iter_jsonl(paths: Sequence[str]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Bad JSONL at {p}:{line_no}: {e}") from e
                if not isinstance(obj, dict):
                    raise ValueError(f"JSONL record must be an object at {p}:{line_no}")
                yield obj


@dataclass(frozen=True)
class DataConfig:
    mode: str  # "cpt" | "sft"
    format: str = "jsonl"  # "jsonl" | "parquet"
    train_files: Tuple[str, ...]
    eval_files: Tuple[str, ...] = ()
    seq_len: int = 2048
    # For M1 we default to *no packing* to avoid cross-sample n-gram pollution.
    # If you enable packing later, also implement boundary handling for Engram.
    packing: bool = False
    text_field: str = "text"
    prompt_field: str = "prompt"
    response_field: str = "response"
    # OpenThoughts-style parquet conversations support.
    conversation_field: str = "conversations"
    conversation_from_field: str = "from"
    conversation_value_field: str = "value"


class CPTDataset(Dataset):
    """Causal LM dataset from raw text records.

    Expected JSONL schema (per line):
      {"text": "..."}
    """

    def __init__(
        self,
        *,
        tokenizer,
        records: List[Dict[str, Any]],
        seq_len: int,
        text_field: str,
    ):
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.text_field = text_field

        self._input_ids: List[List[int]] = []
        for r in records:
            text = r.get(text_field, None)
            if not isinstance(text, str):
                raise ValueError(f"Expected field '{text_field}' to be string, got: {type(text)}")
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if getattr(tokenizer, "eos_token_id", None) is not None:
                ids = ids + [int(tokenizer.eos_token_id)]
            if not ids:
                continue
            self._input_ids.append(ids)

        if not self._input_ids:
            raise ValueError("No usable records found for CPTDataset.")

    def __len__(self) -> int:
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self._input_ids[idx][: self.seq_len]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(ids, dtype=torch.long),
        }


class CPTParquetDataset(Dataset):
    """CPT dataset backed by a HuggingFace `datasets.Dataset` loaded from Parquet.

    Tokenization is on-the-fly to avoid materializing all token ids in RAM.
    """

    def __init__(self, *, tokenizer, hf_dataset, seq_len: int, text_field: str):
        self.tokenizer = tokenizer
        self.ds = hf_dataset
        self.seq_len = int(seq_len)
        self.text_field = str(text_field)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.ds[int(idx)]
        text = r.get(self.text_field, None)
        if isinstance(text, list) and self.text_field == "conversations":
            text = render_conversations_simple(text)
        if not isinstance(text, str):
            raise ValueError(f"Expected field '{self.text_field}' to be string (or conversations list), got: {type(text)}")
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            ids = ids + [int(self.tokenizer.eos_token_id)]
        ids = ids[: self.seq_len]
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "labels": torch.tensor(ids, dtype=torch.long)}


class SFTDataset(Dataset):
    """SFT dataset with prompt/response masking.

    Expected JSONL schema (per line):
      {"prompt": "...", "response": "..."}
    """

    def __init__(
        self,
        *,
        tokenizer,
        records: List[Dict[str, Any]],
        seq_len: int,
        prompt_field: str,
        response_field: str,
    ):
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.prompt_field = prompt_field
        self.response_field = response_field

        self._examples: List[Tuple[List[int], List[int]]] = []
        for r in records:
            prompt = r.get(prompt_field, None)
            response = r.get(response_field, None)
            if not isinstance(prompt, str) or not isinstance(response, str):
                raise ValueError(
                    f"Expected '{prompt_field}'/'{response_field}' to be strings, got: "
                    f"{type(prompt)}/{type(response)}"
                )
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            response_ids = tokenizer(response, add_special_tokens=False).input_ids
            if getattr(tokenizer, "eos_token_id", None) is not None:
                response_ids = response_ids + [int(tokenizer.eos_token_id)]
            if not response_ids:
                continue
            self._examples.append((prompt_ids, response_ids))

        if not self._examples:
            raise ValueError("No usable records found for SFTDataset.")

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt_ids, response_ids = self._examples[idx]
        input_ids = (prompt_ids + response_ids)[: self.seq_len]

        # Mask prompt tokens. Only train on response tokens.
        labels = ([-100] * min(len(prompt_ids), self.seq_len)) + response_ids
        labels = labels[: self.seq_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class SFTParquetDataset(Dataset):
    """SFT dataset backed by Parquet (HF datasets), with prompt/response columns."""

    def __init__(
        self,
        *,
        tokenizer,
        hf_dataset,
        seq_len: int,
        prompt_field: str,
        response_field: str,
        conversation_field: str = "conversations",
        conversation_from_field: str = "from",
        conversation_value_field: str = "value",
    ):
        self.tokenizer = tokenizer
        self.ds = hf_dataset
        self.seq_len = int(seq_len)
        self.prompt_field = str(prompt_field)
        self.response_field = str(response_field)
        self.conversation_field = str(conversation_field)
        self.conversation_from_field = str(conversation_from_field)
        self.conversation_value_field = str(conversation_value_field)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.ds[int(idx)]
        prompt = r.get(self.prompt_field, None)
        response = r.get(self.response_field, None)
        if not isinstance(prompt, str) or not isinstance(response, str):
            # Fallback: OpenThoughts-style conversations.
            conv = r.get(self.conversation_field, None)
            if isinstance(conv, list):
                prompt, response = extract_prompt_response_from_conversations(
                    conv, from_field=self.conversation_from_field, value_field=self.conversation_value_field
                )
            else:
                raise ValueError(
                    f"Expected '{self.prompt_field}'/'{self.response_field}' to be strings, or "
                    f"'{self.conversation_field}' to be a list; got: {type(prompt)}/{type(response)}/{type(conv)}"
                )
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        response_ids = self.tokenizer(response, add_special_tokens=False).input_ids
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            response_ids = response_ids + [int(self.tokenizer.eos_token_id)]
        input_ids = (prompt_ids + response_ids)[: self.seq_len]
        labels = ([-100] * min(len(prompt_ids), self.seq_len)) + response_ids
        labels = labels[: self.seq_len]
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)}


def collate_causal_lm(batch: List[Dict[str, torch.Tensor]], *, pad_token_id: int) -> Dict[str, torch.Tensor]:
    # Dynamic pad to max length in the batch.
    max_len = max(int(x["input_ids"].numel()) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, ex in enumerate(batch):
        ids = ex["input_ids"]
        lbs = ex["labels"]
        L = int(ids.numel())
        input_ids[i, :L] = ids
        labels[i, : int(lbs.numel())] = lbs
        attention_mask[i, :L] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def build_datasets(cfg: DataConfig, *, tokenizer):
    train_files = expand_data_files(cfg.train_files)
    eval_files = expand_data_files(cfg.eval_files) if cfg.eval_files else ()

    if cfg.format == "jsonl":
        train_records = list(iter_jsonl(train_files))
        eval_records = list(iter_jsonl(eval_files)) if eval_files else []
        if cfg.mode == "cpt":
            train_ds = CPTDataset(
                tokenizer=tokenizer, records=train_records, seq_len=cfg.seq_len, text_field=cfg.text_field
            )
            eval_ds = (
                CPTDataset(tokenizer=tokenizer, records=eval_records, seq_len=cfg.seq_len, text_field=cfg.text_field)
                if eval_records
                else None
            )
            return train_ds, eval_ds
        if cfg.mode == "sft":
            train_ds = SFTDataset(
                tokenizer=tokenizer,
                records=train_records,
                seq_len=cfg.seq_len,
                prompt_field=cfg.prompt_field,
                response_field=cfg.response_field,
            )
            eval_ds = (
                SFTDataset(
                    tokenizer=tokenizer,
                    records=eval_records,
                    seq_len=cfg.seq_len,
                    prompt_field=cfg.prompt_field,
                    response_field=cfg.response_field,
                )
                if eval_records
                else None
            )
            return train_ds, eval_ds
        raise ValueError(f"Unknown data mode: {cfg.mode!r} (expected 'cpt' or 'sft')")

    if cfg.format == "parquet":
        try:
            from datasets import load_dataset
        except Exception as e:
            raise RuntimeError(
                "Failed to import `datasets`. For parquet training you need a working parquet backend.\n"
                "If you see a pyarrow parquet error, reinstall a correct-arm64 wheel for your Python.\n"
                "Then `pip install datasets pyarrow` (or use conda-forge)."
            ) from e

        train_hf = load_dataset("parquet", data_files=list(train_files), split="train")
        eval_hf = load_dataset("parquet", data_files=list(eval_files), split="train") if eval_files else None

        if cfg.mode == "cpt":
            return (
                CPTParquetDataset(tokenizer=tokenizer, hf_dataset=train_hf, seq_len=cfg.seq_len, text_field=cfg.text_field),
                CPTParquetDataset(tokenizer=tokenizer, hf_dataset=eval_hf, seq_len=cfg.seq_len, text_field=cfg.text_field)
                if eval_hf is not None
                else None,
            )
        if cfg.mode == "sft":
            return (
                SFTParquetDataset(
                    tokenizer=tokenizer,
                    hf_dataset=train_hf,
                    seq_len=cfg.seq_len,
                    prompt_field=cfg.prompt_field,
                    response_field=cfg.response_field,
                    conversation_field=cfg.conversation_field,
                    conversation_from_field=cfg.conversation_from_field,
                    conversation_value_field=cfg.conversation_value_field,
                ),
                SFTParquetDataset(
                    tokenizer=tokenizer,
                    hf_dataset=eval_hf,
                    seq_len=cfg.seq_len,
                    prompt_field=cfg.prompt_field,
                    response_field=cfg.response_field,
                    conversation_field=cfg.conversation_field,
                    conversation_from_field=cfg.conversation_from_field,
                    conversation_value_field=cfg.conversation_value_field,
                )
                if eval_hf is not None
                else None,
            )
        raise ValueError(f"Unknown data mode: {cfg.mode!r} (expected 'cpt' or 'sft')")

    raise ValueError(f"Unknown data format: {cfg.format!r} (expected 'jsonl' or 'parquet')")

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


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
    train_files: Tuple[str, ...]
    eval_files: Tuple[str, ...] = ()
    seq_len: int = 2048
    # For M1 we default to *no packing* to avoid cross-sample n-gram pollution.
    # If you enable packing later, also implement boundary handling for Engram.
    packing: bool = False
    text_field: str = "text"
    prompt_field: str = "prompt"
    response_field: str = "response"


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
    train_records = list(iter_jsonl(cfg.train_files))
    eval_records = list(iter_jsonl(cfg.eval_files)) if cfg.eval_files else []

    if cfg.mode == "cpt":
        train_ds = CPTDataset(tokenizer=tokenizer, records=train_records, seq_len=cfg.seq_len, text_field=cfg.text_field)
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


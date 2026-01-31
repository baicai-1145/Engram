#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

# Allow running as `python scripts/*.py` without setting PYTHONPATH.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import math
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from src.training.data import DataConfig, build_datasets, collate_causal_lm
from src.training.trainer_utils import load_yaml, make_training_arguments, save_json, seed_everything


def _require_keys(cfg: Dict[str, Any], keys) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="HF checkpoint dir (e.g. runs/... or a saved model dir)")
    ap.add_argument("--config", required=True, help="The same YAML used for training (baseline.yaml)")
    ap.add_argument("--output", default="", help="Write metrics JSON to this path (default: <checkpoint>/eval_metrics.json)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    _require_keys(cfg, ["data"])

    seed = int(cfg.get("seed", 0))
    seed_everything(seed)

    trust_remote_code = bool(cfg.get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=trust_remote_code, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_cfg = DataConfig(
        mode=str(cfg["data"]["mode"]),
        format=str(cfg["data"].get("format", "jsonl")),
        train_files=tuple(cfg["data"]["train_files"]),  # unused, but DataConfig requires it
        eval_files=tuple(cfg["data"].get("eval_files", ())),
        seq_len=int(cfg["data"]["seq_len"]),
        packing=bool(cfg["data"].get("packing", False)),
        text_field=str(cfg["data"].get("text_field", "text")),
        prompt_field=str(cfg["data"].get("prompt_field", "prompt")),
        response_field=str(cfg["data"].get("response_field", "response")),
        conversation_field=str(cfg["data"].get("conversation_field", "conversations")),
        conversation_from_field=str(cfg["data"].get("conversation_from_field", "from")),
        conversation_value_field=str(cfg["data"].get("conversation_value_field", "value")),
        keep_in_memory=bool(cfg["data"].get("keep_in_memory", False)),
        hf_cache_dir=str(cfg["data"].get("hf_cache_dir", "")),
        streaming=bool(cfg["data"].get("streaming", False)),
        shuffle_buffer_size=int(cfg["data"].get("shuffle_buffer_size", 0)),
        streaming_seed=int(cfg["data"].get("streaming_seed", seed)),
    )
    if not data_cfg.eval_files:
        raise ValueError("No eval_files specified in config.data.eval_files")

    _, eval_ds = build_datasets(data_cfg, tokenizer=tokenizer)
    assert eval_ds is not None

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16 if cfg.get("training", {}).get("bf16", True) else None,
    )
    if getattr(model.config, "use_cache", None) is True:
        model.config.use_cache = False

    targs = make_training_arguments(
        output_dir="/tmp/engram_eval_out",
        per_device_eval_batch_size=int(cfg.get("training", {}).get("per_device_eval_batch_size", 1)),
        bf16=bool(cfg.get("training", {}).get("bf16", True)),
        fp16=bool(cfg.get("training", {}).get("fp16", False)),
        dataloader_num_workers=int(cfg.get("training", {}).get("dataloader_num_workers", 2)),
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        eval_dataset=eval_ds,
        data_collator=lambda batch: collate_causal_lm(batch, pad_token_id=int(tokenizer.pad_token_id)),
    )

    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        metrics["eval_ppl"] = float(math.exp(metrics["eval_loss"]))

    out_path = args.output or f"{args.checkpoint.rstrip('/')}/eval_metrics.json"
    save_json(out_path, metrics)
    print(f"Wrote {out_path}")
    if "eval_ppl" in metrics:
        print(f"eval_ppl={metrics['eval_ppl']}")


if __name__ == "__main__":
    main()

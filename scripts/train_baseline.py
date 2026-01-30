#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

# Allow running as `python scripts/*.py` without setting PYTHONPATH.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from src.training.data import DataConfig, build_datasets, collate_causal_lm
from src.training.trainer_utils import create_run_dir, dump_yaml, load_yaml, seed_everything, save_json


def _require_keys(cfg: Dict[str, Any], keys) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")


def _print_data_stats(*, ds, tokenizer, tag: str, max_batches: int = 2) -> None:
    dl = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda batch: collate_causal_lm(batch, pad_token_id=int(tokenizer.pad_token_id)),
    )
    total_tokens = 0
    total_nonpad = 0
    seen = 0
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"]
        attn = batch["attention_mask"]
        total_tokens += int(input_ids.numel())
        total_nonpad += int(attn.sum().item())
        seen += int(input_ids.shape[0])
        print(f"[{tag}] batch{i}: input_ids{tuple(input_ids.shape)} labels{tuple(batch['labels'].shape)}")
    if seen:
        print(f"[{tag}] approx_nonpad_tokens_per_sample={total_nonpad/seen:.1f} (from {seen} samples)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/baseline.yaml)")
    ap.add_argument("--exp_name", default="baseline", help="Experiment name (used in runs/)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    _require_keys(cfg, ["model_name_or_path", "data", "training"])

    seed = int(cfg.get("seed", 0))
    seed_everything(seed)

    run = create_run_dir("runs", args.exp_name, config_path=args.config)
    dump_yaml(run.config_resolved_path, cfg)

    model_name_or_path = cfg["model_name_or_path"]
    trust_remote_code = bool(cfg.get("trust_remote_code", True))

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, use_fast=True)
    if tokenizer.pad_token_id is None:
        # For causal LM fine-tuning, it's common to pad with EOS.
        tokenizer.pad_token = tokenizer.eos_token

    data_cfg = DataConfig(
        mode=str(cfg["data"]["mode"]),
        train_files=tuple(cfg["data"]["train_files"]),
        eval_files=tuple(cfg["data"].get("eval_files", ())),
        seq_len=int(cfg["data"]["seq_len"]),
        packing=bool(cfg["data"].get("packing", False)),
        text_field=str(cfg["data"].get("text_field", "text")),
        prompt_field=str(cfg["data"].get("prompt_field", "prompt")),
        response_field=str(cfg["data"].get("response_field", "response")),
    )

    train_ds, eval_ds = build_datasets(data_cfg, tokenizer=tokenizer)
    _print_data_stats(ds=train_ds, tokenizer=tokenizer, tag="train")
    if eval_ds is not None:
        _print_data_stats(ds=eval_ds, tokenizer=tokenizer, tag="eval", max_batches=1)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16 if cfg["training"].get("bf16", True) else None,
    )

    if cfg["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    # HF Trainer needs this for some causal LMs when gradient checkpointing is on.
    if getattr(model.config, "use_cache", None) is True:
        model.config.use_cache = False

    tcfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=run.run_dir,
        overwrite_output_dir=False,
        per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", tcfg["per_device_train_batch_size"])),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        warmup_steps=int(tcfg.get("warmup_steps", 0)),
        max_steps=int(tcfg["max_steps"]),
        lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "cosine")),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        save_steps=int(tcfg.get("save_steps", 100)),
        eval_steps=int(tcfg.get("eval_steps", 0)) or None,
        evaluation_strategy="steps" if eval_ds is not None and int(tcfg.get("eval_steps", 0)) > 0 else "no",
        save_total_limit=int(tcfg.get("save_total_limit", 2)),
        bf16=bool(tcfg.get("bf16", True)),
        fp16=bool(tcfg.get("fp16", False)),
        dataloader_num_workers=int(tcfg.get("dataloader_num_workers", 2)),
        report_to=[],
        remove_unused_columns=False,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=lambda batch: collate_causal_lm(batch, pad_token_id=int(tokenizer.pad_token_id)),
        tokenizer=tokenizer,
    )

    train_result = trainer.train()
    trainer.save_model()  # saves to output_dir
    tokenizer.save_pretrained(run.run_dir)

    metrics = dict(train_result.metrics)
    if eval_ds is not None:
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval/{k}": v for k, v in eval_metrics.items()})
        if "eval/loss" in metrics:
            metrics["eval/ppl"] = float(math.exp(metrics["eval/loss"]))
        elif "eval/eval_loss" in metrics:
            metrics["eval/ppl"] = float(math.exp(metrics["eval/eval_loss"]))

    save_json(run.metrics_path, metrics)

    # Minimal console output for quick sanity check.
    print(f"Run dir: {run.run_dir}")
    if "train_loss" in metrics:
        print(f"train_loss={metrics['train_loss']}")
    if "eval/ppl" in metrics:
        print(f"eval_ppl={metrics['eval/ppl']}")


if __name__ == "__main__":
    main()

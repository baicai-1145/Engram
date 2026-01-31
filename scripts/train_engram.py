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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_scheduler,
)

from src.engram.config import EngramConfig, normalize_layers
from src.engram.param_groups import collect_engram_runtime_info, build_optimizer_param_groups
from src.engram.patch_qwen import patch_model_with_engram
from src.training.data import DataConfig, build_datasets, collate_causal_lm
from src.training.trainer_utils import (
    create_run_dir,
    dump_yaml,
    load_yaml,
    make_training_arguments,
    save_json,
    seed_everything,
)


def _require_keys(cfg: Dict[str, Any], keys) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")


class EngramTrainer(Trainer):
    def __init__(self, *args, optimizer_groups=None, scheduler_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._optimizer_groups = optimizer_groups
        self._scheduler_cfg = scheduler_cfg or {}

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        if self._optimizer_groups is None:
            return super().create_optimizer()
        self.optimizer = torch.optim.AdamW(
            self._optimizer_groups,
            betas=tuple(self._scheduler_cfg.get("betas", (0.9, 0.95))),
            eps=float(self._scheduler_cfg.get("eps", 1e-8)),
        )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        if optimizer is None:
            optimizer = self.optimizer
        name = str(self._scheduler_cfg.get("lr_scheduler_type", "cosine"))
        warmup_steps = int(self._scheduler_cfg.get("warmup_steps", 0))
        self.lr_scheduler = get_scheduler(
            name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        return self.lr_scheduler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/engram.yaml)")
    ap.add_argument("--exp_name", default="engram", help="Experiment name (used in runs/)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    _require_keys(cfg, ["model_name_or_path", "data", "training", "engram"])

    seed = int(cfg.get("seed", 0))
    seed_everything(seed)

    run = create_run_dir("runs", args.exp_name, config_path=args.config)
    dump_yaml(run.config_resolved_path, cfg)

    model_name_or_path = cfg["model_name_or_path"]
    trust_remote_code = bool(cfg.get("trust_remote_code", True))

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_cfg = DataConfig(
        mode=str(cfg["data"]["mode"]),
        format=str(cfg["data"].get("format", "jsonl")),
        train_files=tuple(cfg["data"]["train_files"]),
        eval_files=tuple(cfg["data"].get("eval_files", ())),
        seq_len=int(cfg["data"]["seq_len"]),
        packing=bool(cfg["data"].get("packing", False)),
        text_field=str(cfg["data"].get("text_field", "text")),
        prompt_field=str(cfg["data"].get("prompt_field", "prompt")),
        response_field=str(cfg["data"].get("response_field", "response")),
        conversation_field=str(cfg["data"].get("conversation_field", "conversations")),
        conversation_from_field=str(cfg["data"].get("conversation_from_field", "from")),
        conversation_value_field=str(cfg["data"].get("conversation_value_field", "value")),
    )
    train_ds, eval_ds = build_datasets(data_cfg, tokenizer=tokenizer)

    tcfg = cfg["training"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16 if tcfg.get("bf16", True) else None,
    )

    # Patch Engram.
    ecfg_raw = cfg["engram"]
    engram_cfg = EngramConfig(
        enabled=bool(ecfg_raw.get("enabled", True)),
        code_layers=tuple(normalize_layers(ecfg_raw["layers"], layer_index_base=int(ecfg_raw.get("layer_index_base", 0)))),
        layer_index_base=int(ecfg_raw.get("layer_index_base", 0)),
        max_ngram=int(ecfg_raw.get("max_ngram", 3)),
        num_heads=int(ecfg_raw.get("num_heads", 8)),
        table_size_per_ngram=tuple(int(x) for x in ecfg_raw.get("table_size_per_ngram", [])),
        pad_token_id=ecfg_raw.get("pad_token_id", None),
        seed=int(ecfg_raw.get("seed", seed)),
        d_mem=ecfg_raw.get("d_mem", None),
        kernel_size=int(ecfg_raw.get("kernel_size", 4)),
        cache_dir=str(ecfg_raw.get("cache_dir", ".cache/engram")),
        tokenizer_compression_mode=str(ecfg_raw.get("tokenizer_compression_mode", "demo")),
        gating_mode=str(ecfg_raw.get("gating_mode", "paper")),
        init_equivalence=str(ecfg_raw.get("init_equivalence", "zero_output")),
    )

    patched = patch_model_with_engram(model, cfg=engram_cfg, tokenizer=tokenizer)
    print(f"[engram] patched_blocks={patched.num_patched} layers={patched.code_layers} block_path={patched.block_path}")

    # Optimizer param groups (Engram lr*5 wd=0).
    groups, summaries = build_optimizer_param_groups(
        model,
        base_lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        engram_lr_scale=float(ecfg_raw.get("lr_scale", 5.0)),
    )
    for s in summaries:
        print(f"[optim] {s.name}: params={s.num_params} lr={s.lr} wd={s.weight_decay}")

    # Print Engram runtime info (table sizes, embed params, compression ratio).
    infos = collect_engram_runtime_info(model)
    if infos:
        total_embed_params = sum(x["embed_params"] for x in infos)
        bytes_per_param = 2 if bool(tcfg.get("bf16", True)) or bool(tcfg.get("fp16", False)) else 4
        print(f"[engram] total_embed_params={total_embed_params} (~{total_embed_params*bytes_per_param/1e9:.2f} GB in weights)")
        for x in sorted(infos, key=lambda d: d["layer_id"]):
            ratio = 1.0 - (x["new_vocab_size"] / max(1, x["old_vocab_size"]))
            print(
                f"[engram] layer={x['layer_id']} total_rows={x['total_rows']} d_head={x['d_head']} "
                f"embed_params={x['embed_params']} compression_reduction={ratio*100:.1f}%"
            )

    if tcfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    if getattr(model.config, "use_cache", None) is True:
        model.config.use_cache = False

    training_args = make_training_arguments(
        output_dir=run.run_dir,
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

    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=lambda batch: collate_causal_lm(batch, pad_token_id=int(tokenizer.pad_token_id)),
        tokenizer=tokenizer,
        optimizer_groups=groups,
        scheduler_cfg={
            "lr_scheduler_type": tcfg.get("lr_scheduler_type", "cosine"),
            "warmup_steps": tcfg.get("warmup_steps", 0),
            "betas": tcfg.get("betas", (0.9, 0.95)),
            "eps": tcfg.get("adam_eps", 1e-8),
        },
    )

    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(run.run_dir)

    metrics = dict(train_result.metrics)
    if eval_ds is not None:
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval/{k}": v for k, v in eval_metrics.items()})
        if "eval/eval_loss" in metrics:
            metrics["eval/ppl"] = float(math.exp(metrics["eval/eval_loss"]))

    save_json(run.metrics_path, metrics)
    print(f"Run dir: {run.run_dir}")
    if "eval/ppl" in metrics:
        print(f"eval_ppl={metrics['eval/ppl']}")


if __name__ == "__main__":
    main()

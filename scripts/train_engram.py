#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

# Allow running as `python scripts/*.py` without setting PYTHONPATH.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import math
import time
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
from src.engram.module import EngramLayer
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
    def __init__(
        self,
        *args,
        optimizer_groups=None,
        scheduler_cfg=None,
        log_engram_stats: bool = False,
        engram_stats_sample_tokens: int = 4096,
        weight_stats_every: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._optimizer_groups = optimizer_groups
        self._scheduler_cfg = scheduler_cfg or {}
        self._perf_tokens_since_step = 0
        self._perf_examples_since_step = 0
        self._perf_microsteps_since_step = 0
        self._perf_last_opt_end = time.perf_counter()
        self._last_step_metrics: Dict[str, float] = {}
        self._weight_stats_every = int(weight_stats_every)
        self._log_engram_stats = bool(log_engram_stats)
        self._engram_stats_sample_tokens = int(engram_stats_sample_tokens)
        self._logged_constant_stats = False

    def log(self, logs: Dict[str, float], start_time: float | None = None) -> None:
        # Add per-param-group LR to logs (useful for Engram's lr_scale).
        if self.optimizer is not None:
            for i, g in enumerate(self.optimizer.param_groups):
                name = str(g.get("name", f"group{i}"))
                lr = g.get("lr", None)
                if lr is not None:
                    logs[f"lr/{name}"] = float(lr)

        # Log GPU memory to help diagnose "why batch size won't go up".
        if torch.cuda.is_available():
            try:
                logs["gpu/mem_alloc_mb"] = float(torch.cuda.memory_allocated() / 1024 / 1024)
                logs["gpu/mem_reserved_mb"] = float(torch.cuda.memory_reserved() / 1024 / 1024)
                logs["gpu/mem_alloc_max_mb"] = float(torch.cuda.max_memory_allocated() / 1024 / 1024)
                logs["gpu/mem_reserved_max_mb"] = float(torch.cuda.max_memory_reserved() / 1024 / 1024)
            except Exception:
                pass

        # Perf + grad stats recorded at optimizer_step().
        for k, v in self._last_step_metrics.items():
            logs.setdefault(k, float(v))

        # Engram forward stats (per patched layer).
        if self._log_engram_stats:
            for m in self.model.modules():
                if not isinstance(m, EngramLayer):
                    continue
                lid = int(getattr(m, "layer_id", -1))
                stats = getattr(m, "last_stats", None)
                if not stats:
                    continue
                for sk, sv in stats.items():
                    logs.setdefault(f"engram/l{lid}/{sk}", float(sv))

        # One-time Engram constants (useful to correlate curves with table size / params).
        if (not self._logged_constant_stats) and self.state.global_step >= 0:
            self._logged_constant_stats = True
            for m in self.model.modules():
                if not isinstance(m, EngramLayer):
                    continue
                lid = int(getattr(m, "layer_id", -1))
                ri = getattr(m, "runtime_info", None)
                if ri is None:
                    continue
                logs.setdefault(f"engram/l{lid}/total_rows", float(getattr(ri, "total_rows", 0)))
                logs.setdefault(f"engram/l{lid}/embed_params", float(getattr(ri, "embed_params", 0)))
                logs.setdefault(f"engram/l{lid}/d_mem", float(getattr(ri, "d_mem", 0)))
                logs.setdefault(f"engram/l{lid}/d_head", float(getattr(ri, "d_head", 0)))

        # Optional: weight norms (expensive for huge embeddings). Run at a low frequency.
        ws_every = int(self._weight_stats_every)
        if ws_every > 0 and self.state.global_step > 0 and (self.state.global_step % ws_every == 0):
            with torch.no_grad():
                for m in self.model.modules():
                    if not isinstance(m, EngramLayer):
                        continue
                    lid = int(getattr(m, "layer_id", -1))
                    try:
                        conv_w = m.short_conv.conv.weight.detach().float()
                        key_w = m.key_proj.weight.detach().float()
                        val_w = m.value_proj.weight.detach().float()
                        qn_w = m.q_norm.weight.detach().float()
                        kn_w = m.k_norm.weight.detach().float()
                        emb_w = m.memory_embedding.embedding.weight.detach().float()

                        logs[f"engram/l{lid}/conv_w_norm"] = float(conv_w.norm().item())
                        logs[f"engram/l{lid}/conv_w_rms"] = float(conv_w.pow(2).mean().sqrt().item())
                        logs[f"engram/l{lid}/key_w_norm"] = float(key_w.norm().item())
                        logs[f"engram/l{lid}/key_w_rms"] = float(key_w.pow(2).mean().sqrt().item())
                        logs[f"engram/l{lid}/value_w_norm"] = float(val_w.norm().item())
                        logs[f"engram/l{lid}/value_w_rms"] = float(val_w.pow(2).mean().sqrt().item())
                        logs[f"engram/l{lid}/qnorm_w_norm"] = float(qn_w.norm().item())
                        logs[f"engram/l{lid}/qnorm_w_rms"] = float(qn_w.pow(2).mean().sqrt().item())
                        logs[f"engram/l{lid}/knorm_w_norm"] = float(kn_w.norm().item())
                        logs[f"engram/l{lid}/knorm_w_rms"] = float(kn_w.pow(2).mean().sqrt().item())
                        # Potentially huge: embedding weight.
                        logs[f"engram/l{lid}/embed_w_norm"] = float(emb_w.norm().item())
                        logs[f"engram/l{lid}/embed_w_rms"] = float(emb_w.pow(2).mean().sqrt().item())
                    except Exception:
                        # Never fail training due to optional logging.
                        pass

        return super().log(logs, start_time=start_time)

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Track throughput. Tokens/examples are counted on the microbatch level and
        # flushed once per optimizer step.
        try:
            ids = inputs.get("input_ids", None)
            if ids is not None:
                self._perf_tokens_since_step += int(ids.numel())
                self._perf_examples_since_step += int(ids.shape[0])
                self._perf_microsteps_since_step += 1
        except Exception:
            pass

        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def optimizer_step(self, *args, **kwargs):
        # Called once per optimizer update (after gradient accumulation).
        now = time.perf_counter()
        dt = max(1e-9, now - self._perf_last_opt_end)
        tokens = int(self._perf_tokens_since_step)
        examples = int(self._perf_examples_since_step)
        microsteps = int(self._perf_microsteps_since_step)
        self._perf_tokens_since_step = 0
        self._perf_examples_since_step = 0
        self._perf_microsteps_since_step = 0

        step_metrics: Dict[str, float] = {
            "perf/step_time_sec": float(dt),
            "perf/tokens_per_sec": float(tokens / dt) if tokens > 0 else 0.0,
            "perf/examples_per_sec": float(examples / dt) if examples > 0 else 0.0,
            "perf/microsteps_per_opt_step": float(microsteps),
        }

        # Grad norms are expensive; compute only at logging cadence.
        do_grad_stats = (self.args.logging_steps > 0) and (
            self.state.global_step == 0 or (self.state.global_step % int(self.args.logging_steps) == 0)
        )
        if do_grad_stats and self.optimizer is not None:
            for i, g in enumerate(self.optimizer.param_groups):
                name = str(g.get("name", f"group{i}"))
                sq = None
                psq = None
                for p in g.get("params", []):
                    grad = getattr(p, "grad", None)
                    if grad is None:
                        continue
                    v = grad.detach()
                    if v.is_sparse:
                        v = v.coalesce().values()
                    s = v.float().pow(2).sum()
                    sq = s if sq is None else (sq + s)
                    # Param norm for the same group (tracked alongside grad norm).
                    w = p.detach()
                    if w.is_sparse:
                        w = w.coalesce().values()
                    ws = w.float().pow(2).sum()
                    psq = ws if psq is None else (psq + ws)
                if sq is not None:
                    step_metrics[f"optim/grad_norm/{name}"] = float(torch.sqrt(sq).item())
                if psq is not None:
                    step_metrics[f"optim/param_norm/{name}"] = float(torch.sqrt(psq).item())

        self._last_step_metrics = step_metrics
        out = super().optimizer_step(*args, **kwargs)
        self._perf_last_opt_end = time.perf_counter()
        return out

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
        keep_in_memory=bool(cfg["data"].get("keep_in_memory", False)),
        hf_cache_dir=str(cfg["data"].get("hf_cache_dir", "")),
        streaming=bool(cfg["data"].get("streaming", False)),
        shuffle_buffer_size=int(cfg["data"].get("shuffle_buffer_size", 0)),
        streaming_seed=int(cfg["data"].get("streaming_seed", seed)),
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

    # Optional ablation / speedup: freeze the backbone and train only Engram params.
    # This reduces optimizer state + weight-grad compute, but still backprops through later layers
    # to update Engram (so it won't be a 10x speedup).
    if bool(tcfg.get("train_only_engram", False)):
        trainable = 0
        frozen = 0
        for name, p in model.named_parameters():
            if ".engram." in name:
                p.requires_grad_(True)
                trainable += p.numel()
            else:
                if p.requires_grad:
                    p.requires_grad_(False)
                frozen += p.numel()
        total = trainable + frozen
        pct = 100.0 * (trainable / max(1, total))
        print(f"[train] train_only_engram=true trainable_params={trainable} ({pct:.2f}%) frozen_params={frozen}")

    # Enable Engram internal stat collection for visualization (TensorBoard).
    if bool(tcfg.get("log_engram_stats", False)):
        sample_n = int(tcfg.get("engram_stats_sample_tokens", 4096))
        for m in model.modules():
            if isinstance(m, EngramLayer):
                m.collect_stats = True
                m.stats_sample_tokens = sample_n
        print(f"[engram] log_engram_stats=true sample_tokens={sample_n}")

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
        save_only_model=bool(tcfg.get("save_only_model", False)),
        bf16=bool(tcfg.get("bf16", True)),
        fp16=bool(tcfg.get("fp16", False)),
        dataloader_num_workers=int(tcfg.get("dataloader_num_workers", 2)),
        dataloader_pin_memory=bool(tcfg.get("dataloader_pin_memory", True)),
        dataloader_persistent_workers=bool(tcfg.get("dataloader_persistent_workers", True)),
        dataloader_prefetch_factor=int(tcfg.get("dataloader_prefetch_factor", 4)),
        report_to=list(tcfg.get("report_to", [])),
        logging_dir=str(tcfg.get("logging_dir", os.path.join(run.run_dir, "tb"))),
        remove_unused_columns=False,
        seed=seed,
    )

    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=lambda batch: collate_causal_lm(batch, pad_token_id=int(tokenizer.pad_token_id)),
        optimizer_groups=groups,
        log_engram_stats=bool(tcfg.get("log_engram_stats", False)),
        engram_stats_sample_tokens=int(tcfg.get("engram_stats_sample_tokens", 4096)),
        weight_stats_every=int(tcfg.get("weight_stats_every", 0)),
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

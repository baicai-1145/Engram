#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

# Allow running as `python scripts/*.py` without setting PYTHONPATH.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.engram.config import EngramConfig, normalize_layers
from src.engram.patch_qwen import patch_model_with_engram
from src.training.trainer_utils import load_yaml, seed_everything


def _require_keys(cfg: Dict[str, Any], keys) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="T2.6: baseline vs injected Engram forward should be (near) identical.")
    ap.add_argument("--config", required=True, help="configs/engram.yaml (or equivalent)")
    ap.add_argument("--text", default="Hello world!", help="Probe text")
    ap.add_argument("--device", default="cpu", help="cpu|cuda")
    ap.add_argument("--max_abs_diff", type=float, default=0.0, help="Fail if max abs diff > this")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    _require_keys(cfg, ["model_name_or_path", "engram"])

    seed = int(cfg.get("seed", 0))
    seed_everything(seed)

    model_name_or_path = cfg["model_name_or_path"]
    trust_remote_code = bool(cfg.get("trust_remote_code", True))

    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    batch = tok(args.text, return_tensors="pt")
    batch = {k: v.to(args.device) for k, v in batch.items()}

    # Baseline model
    base = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code).to(args.device)
    base.eval()
    out_base = base(**batch).logits

    # Patched model (same weights)
    patched = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code).to(args.device)
    patched.eval()

    ecfg_raw = cfg["engram"]
    engram_cfg = EngramConfig(
        enabled=True,
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

    patch_model_with_engram(patched, cfg=engram_cfg, tokenizer=tok)
    out_pat = patched(**batch).logits

    diff = (out_base - out_pat).abs()
    max_abs = float(diff.max().item())
    print(f"max_abs_diff={max_abs}")
    if max_abs > float(args.max_abs_diff):
        raise SystemExit(f"FAILED: max_abs_diff {max_abs} > threshold {args.max_abs_diff}")
    print("OK")


if __name__ == "__main__":
    main()

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


def _pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device)


def _pick_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    m = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype not in m:
        raise ValueError(f"Unknown dtype {dtype!r}, choose from: auto|bf16|fp16|fp32")
    return m[dtype]


def _load_state_dict(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    st_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(st_path):
        try:
            from safetensors.torch import load_file
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Found model.safetensors but failed to import safetensors. "
                "Install it with: pip install safetensors"
            ) from e
        return load_file(st_path, device="cpu")

    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(bin_path):
        obj = torch.load(bin_path, map_location="cpu")
        if not isinstance(obj, dict):
            raise RuntimeError(f"Unexpected pytorch_model.bin type: {type(obj)}")
        return obj

    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found under: {checkpoint_dir}")


def _load_engram_only_state_dict(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """Load only `.engram.` tensors from a HF checkpoint.

    We intentionally avoid loading the full (large) state dict when possible and
    avoid strict loading issues with tied weights (e.g. lm_head).
    """
    st_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(st_path):
        try:
            from safetensors import safe_open
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Found model.safetensors but failed to import safetensors. Install it with: pip install safetensors"
            ) from e
        out: Dict[str, torch.Tensor] = {}
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if ".engram." in k:
                    out[k] = f.get_tensor(k)
        return out

    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(bin_path):
        obj = torch.load(bin_path, map_location="cpu")
        if not isinstance(obj, dict):
            raise RuntimeError(f"Unexpected pytorch_model.bin type: {type(obj)}")
        return {k: v for k, v in obj.items() if ".engram." in k}

    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found under: {checkpoint_dir}")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="e.g. checkpoint-4800/")
    ap.add_argument("--config", required=True, help="Training YAML (for engram config + base model path)")
    ap.add_argument("--prompt", default="Hello!", help="Raw prompt text")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--dtype", default="auto", help="auto|bf16|fp16|fp32")
    ap.add_argument(
        "--use_cache",
        action="store_true",
        help="Enable KV cache for faster generation. WARNING: Engram will only see the last token each step.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--chat",
        action="store_true",
        help="Use tokenizer.apply_chat_template if available (single user message).",
    )
    ap.add_argument(
        "--no_strict",
        action="store_true",
        help="Load checkpoint with strict=False (not recommended; may silently drop Engram weights on mismatch).",
    )
    args = ap.parse_args()

    seed_everything(int(args.seed))

    cfg = load_yaml(args.config)
    _require_keys(cfg, ["model_name_or_path"])
    model_name_or_path = str(cfg["model_name_or_path"])
    trust_remote_code = bool(cfg.get("trust_remote_code", True))

    device = _pick_device(str(args.device))
    dtype = _pick_dtype(str(args.dtype), device)

    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Load the trained backbone weights from the checkpoint dir using transformers, so tied weights
    # (e.g. lm_head) are handled correctly. Engram weights are ignored at this stage.
    model = AutoModelForCausalLM.from_pretrained(
        str(args.checkpoint),
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype if device.type in ("cuda", "mps") else None,
    )

    if "engram" in cfg:
        ecfg_raw = cfg["engram"]
        engram_cfg = EngramConfig(
            enabled=bool(ecfg_raw.get("enabled", True)),
            code_layers=tuple(
                normalize_layers(ecfg_raw["layers"], layer_index_base=int(ecfg_raw.get("layer_index_base", 0)))
            ),
            layer_index_base=int(ecfg_raw.get("layer_index_base", 0)),
            max_ngram=int(ecfg_raw.get("max_ngram", 3)),
            num_heads=int(ecfg_raw.get("num_heads", 8)),
            table_size_per_ngram=tuple(int(x) for x in ecfg_raw.get("table_size_per_ngram", [])),
            pad_token_id=ecfg_raw.get("pad_token_id", None),
            seed=int(ecfg_raw.get("seed", int(args.seed))),
            d_mem=ecfg_raw.get("d_mem", None),
            kernel_size=int(ecfg_raw.get("kernel_size", 4)),
            cache_dir=str(ecfg_raw.get("cache_dir", ".cache/engram")),
            tokenizer_compression_mode=str(ecfg_raw.get("tokenizer_compression_mode", "demo")),
            gating_mode=str(ecfg_raw.get("gating_mode", "paper")),
            init_equivalence=str(ecfg_raw.get("init_equivalence", "zero_output")),
        )
        patched = patch_model_with_engram(model, cfg=engram_cfg, tokenizer=tok)
        print(f"[engram] patched_blocks={patched.num_patched} layers={patched.code_layers} block_path={patched.block_path}")

    # Load only Engram weights into the now-patched model.
    if "engram" in cfg:
        engram_sd = _load_engram_only_state_dict(str(args.checkpoint))
        strict = not bool(args.no_strict)
        missing, unexpected = model.load_state_dict(engram_sd, strict=False)
        missing_engram = [k for k in missing if ".engram." in k]
        unexpected_engram = [k for k in unexpected if ".engram." in k]
        if missing_engram:
            print(f"[warn] missing_engram_keys={len(missing_engram)} (showing first 20): {missing_engram[:20]}")
        if unexpected_engram:
            print(f"[warn] unexpected_engram_keys={len(unexpected_engram)} (showing first 20): {unexpected_engram[:20]}")
        if strict and (missing_engram or unexpected_engram):
            raise RuntimeError("Engram weights mismatch. Rerun with the exact training YAML, or pass --no_strict.")

    model.to(device)
    model.eval()
    model.config.use_cache = bool(args.use_cache)

    if args.chat and hasattr(tok, "apply_chat_template"):
        messages = [{"role": "user", "content": str(args.prompt)}]
        input_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        input_ids = input_ids.to(device)
    else:
        batch = tok(str(args.prompt), return_tensors="pt")
        input_ids = batch["input_ids"].to(device)

    gen = model.generate(
        input_ids=input_ids,
        max_new_tokens=int(args.max_new_tokens),
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        use_cache=bool(args.use_cache),
        pad_token_id=int(tok.pad_token_id),
        eos_token_id=int(tok.eos_token_id),
    )
    text = tok.decode(gen[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()

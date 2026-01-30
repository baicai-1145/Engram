from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.engram.config import EngramConfig
from src.engram.module import EngramLayer


@dataclass(frozen=True)
class PatchedInfo:
    code_layers: Tuple[int, ...]
    block_path: str
    num_patched: int


def _find_blocks(model: nn.Module) -> tuple[list[nn.Module], str]:
    # Best-effort: try common HF layouts.
    candidates = [
        ("model.layers", ["model", "layers"]),
        ("model.model.layers", ["model", "model", "layers"]),
        ("transformer.h", ["transformer", "h"]),
        ("gpt_neox.layers", ["gpt_neox", "layers"]),
        ("backbone.layers", ["backbone", "layers"]),
        ("layers", ["layers"]),
    ]
    for path_str, path in candidates:
        cur: Any = model
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if not ok:
            continue
        if isinstance(cur, (nn.ModuleList, list, tuple)):
            blocks = list(cur)
            if blocks and all(isinstance(b, nn.Module) for b in blocks):
                return blocks, path_str
    raise RuntimeError(
        "Cannot locate transformer blocks. Tried common paths like model.layers / transformer.h. "
        "Inspect your model class and extend _find_blocks()."
    )


def patch_model_with_engram(model: nn.Module, *, cfg: EngramConfig, tokenizer) -> PatchedInfo:
    """Inject Engram into specific transformer blocks using forward_pre_hook.

    Implementation detail:
    - We stash input_ids on the model object during model.forward().
    - Each patched block reads model._engram_input_ids during its pre-hook.
    """
    if not cfg.enabled:
        return PatchedInfo(code_layers=tuple(), block_path="", num_patched=0)

    blocks, block_path = _find_blocks(model)
    num_layers = len(blocks)

    code_layers = list(cfg.code_layers)
    for lid in code_layers:
        if lid < 0 or lid >= num_layers:
            raise ValueError(f"Invalid layer id {lid} for num_layers={num_layers}")

    # Patch model.forward to stash input_ids.
    if not hasattr(model, "_engram_forward_patched"):
        orig_forward = model.forward

        def wrapped_forward(*args, **kwargs):
            if "input_ids" in kwargs and kwargs["input_ids"] is not None:
                model._engram_input_ids = kwargs["input_ids"]
            elif args:
                # Some models call forward(input_ids, ...)
                model._engram_input_ids = args[0]
            return orig_forward(*args, **kwargs)

        model.forward = wrapped_forward  # type: ignore[assignment]
        model._engram_forward_patched = True  # type: ignore[attr-defined]

    # Resolve hidden_size from model config if possible.
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(getattr(model, "config", None), "n_embd", None)
    if hidden_size is None:
        raise RuntimeError("Cannot infer hidden_size from model.config (expected hidden_size or n_embd).")
    hidden_size = int(hidden_size)

    # Build Engram module per layer (demo-style: per-layer tables).
    table_sizes = cfg.table_size_per_ngram
    if not table_sizes:
        # Default base sizes: smallish for single-card sanity; user should override in YAML.
        table_sizes = tuple([500_000] * (cfg.max_ngram - 1))

    raw_pad_id = cfg.pad_token_id
    if raw_pad_id is None:
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id is None; set cfg.pad_token_id explicitly.")
        raw_pad_id = int(tokenizer.pad_token_id)

    patched = 0
    # Keep Engram modules in the same dtype as the backbone to avoid mixed-dtype RMSNorm slow paths.
    try:
        backbone_dtype = next(model.parameters()).dtype
    except StopIteration:
        backbone_dtype = torch.float32

    for lid in code_layers:
        block = blocks[lid]
        if hasattr(block, "engram"):
            raise RuntimeError(f"Block {lid} already has attribute 'engram' - refusing to overwrite.")

        block.engram = EngramLayer(  # type: ignore[attr-defined]
            layer_id=lid,
            tokenizer=tokenizer,
            hidden_size=hidden_size,
            max_ngram=cfg.max_ngram,
            num_heads=cfg.num_heads,
            table_size_per_ngram=table_sizes,
            pad_token_id=int(raw_pad_id),
            seed=cfg.seed,
            kernel_size=cfg.kernel_size,
            d_mem=cfg.d_mem,
            tokenizer_compression_mode=cfg.tokenizer_compression_mode,
            gating_mode=cfg.gating_mode,
            init_equivalence=cfg.init_equivalence,
            cache_dir=cfg.cache_dir,
        )
        block.engram.to(dtype=backbone_dtype)  # type: ignore[attr-defined]

        def _make_prehook(model_ref: nn.Module):
            def prehook(module: nn.Module, args, kwargs):
                if not hasattr(module, "engram"):
                    return args, kwargs
                if "hidden_states" in kwargs and kwargs["hidden_states"] is not None:
                    hs = kwargs["hidden_states"]
                    use_kwargs = True
                elif args:
                    hs = args[0]
                    use_kwargs = False
                else:
                    return args, kwargs

                input_ids = getattr(model_ref, "_engram_input_ids", None)
                if input_ids is None:
                    return args, kwargs

                y = module.engram(hidden_states=hs, input_ids=input_ids)  # type: ignore[attr-defined]
                hs2 = hs + y.to(dtype=hs.dtype)

                if use_kwargs:
                    kwargs = dict(kwargs)
                    kwargs["hidden_states"] = hs2
                    return args, kwargs
                return (hs2,) + tuple(args[1:]), kwargs

            return prehook

        block.register_forward_pre_hook(_make_prehook(model), with_kwargs=True)
        patched += 1

    return PatchedInfo(code_layers=tuple(code_layers), block_path=block_path, num_patched=patched)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from src.engram.module import EngramLayer


@dataclass(frozen=True)
class ParamGroupSummary:
    name: str
    num_params: int
    lr: float
    weight_decay: float


def _is_norm_param(name: str, module: nn.Module) -> bool:
    # Best-effort: match common norm parameter names.
    if name.endswith(".bias"):
        return True
    if any(x in name.lower() for x in ("layernorm", "rmsnorm", "norm")) and name.endswith(".weight"):
        return True
    # Also treat embedding weights as decay-free by default.
    if ".embed_tokens." in name or name.endswith("embed_tokens.weight"):
        return True
    return False


def build_optimizer_param_groups(
    model: nn.Module,
    *,
    base_lr: float,
    weight_decay: float,
    engram_lr_scale: float = 5.0,
) -> Tuple[List[Dict], List[ParamGroupSummary]]:
    """Create param groups:
    - base_decay / base_no_decay
    - engram (all) with lr scaled and wd=0 (paper: embedding + WK/WV with lr*5, wd=0)
    """
    base_decay: List[torch.nn.Parameter] = []
    base_no_decay: List[torch.nn.Parameter] = []
    engram_fast: List[torch.nn.Parameter] = []  # embedding + (optional) WK/WV
    engram_slow: List[torch.nn.Parameter] = []  # conv/norm/etc.

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Anything under an EngramLayer goes to engram groups.
        if ".engram." in name:
            # Paper/plan: embedding + WK/WV use lr*5 and wd=0.
            if name.endswith("memory_embedding.embedding.weight") or ".key_proj." in name or ".value_proj." in name:
                engram_fast.append(p)
            else:
                engram_slow.append(p)
            continue

        if _is_norm_param(name, model):
            base_no_decay.append(p)
        else:
            base_decay.append(p)

    groups: List[Dict] = []
    summaries: List[ParamGroupSummary] = []

    if base_decay:
        groups.append({"params": base_decay, "lr": float(base_lr), "weight_decay": float(weight_decay)})
        summaries.append(
            ParamGroupSummary(
                name="base_decay",
                num_params=sum(p.numel() for p in base_decay),
                lr=float(base_lr),
                weight_decay=float(weight_decay),
            )
        )
    if base_no_decay:
        groups.append({"params": base_no_decay, "lr": float(base_lr), "weight_decay": 0.0})
        summaries.append(
            ParamGroupSummary(
                name="base_no_decay",
                num_params=sum(p.numel() for p in base_no_decay),
                lr=float(base_lr),
                weight_decay=0.0,
            )
        )
    if engram_fast:
        groups.append({"params": engram_fast, "lr": float(base_lr) * float(engram_lr_scale), "weight_decay": 0.0})
        summaries.append(
            ParamGroupSummary(
                name="engram_fast",
                num_params=sum(p.numel() for p in engram_fast),
                lr=float(base_lr) * float(engram_lr_scale),
                weight_decay=0.0,
            )
        )
    if engram_slow:
        groups.append({"params": engram_slow, "lr": float(base_lr), "weight_decay": 0.0})
        summaries.append(
            ParamGroupSummary(
                name="engram_slow",
                num_params=sum(p.numel() for p in engram_slow),
                lr=float(base_lr),
                weight_decay=0.0,
            )
        )

    return groups, summaries


def collect_engram_runtime_info(model: nn.Module) -> List[dict]:
    infos = []
    for m in model.modules():
        if isinstance(m, EngramLayer):
            ri = getattr(m, "runtime_info", None)
            if ri is None:
                continue
            infos.append(
                {
                    "layer_id": int(m.layer_id),
                    "old_vocab_size": int(ri.old_vocab_size),
                    "new_vocab_size": int(ri.new_vocab_size),
                    "num_heads_total": int(ri.num_heads_total),
                    "d_head": int(ri.d_head),
                    "d_mem": int(ri.d_mem),
                    "total_rows": int(ri.total_rows),
                    "embed_params": int(ri.embed_params),
                }
            )
    return infos

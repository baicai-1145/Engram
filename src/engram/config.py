from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class EngramConfig:
    # Core on/off + placement.
    enabled: bool = False
    code_layers: Tuple[int, ...] = ()
    layer_index_base: int = 0  # 0 => code_layers are 0-based; 1 => convert from 1-based

    # Retrieval.
    max_ngram: int = 3  # use {2..max_ngram}-grams
    num_heads: int = 8  # K hash heads per n-gram order
    table_size_per_ngram: Tuple[int, ...] = ()  # length = max_ngram-1, base sizes before prime search
    pad_token_id: Optional[int] = None  # raw pad id (pre-compression). If None, will use tokenizer.pad_token_id.
    seed: int = 0

    # Embedding/feature sizes.
    d_mem: Optional[int] = None  # if None, defaults to hidden_size//2
    kernel_size: int = 4

    # Engineering.
    cache_dir: str = ".cache/engram"

    # Alignment decisions (D6/D7).
    tokenizer_compression_mode: str = "demo"  # "demo" | "paper"
    gating_mode: str = "paper"  # "paper" | "demo"
    init_equivalence: str = "zero_output"  # "zero_output" | "none"


def normalize_layers(layers: Iterable[int], *, layer_index_base: int) -> tuple[int, ...]:
    """Convert user-specified layers into 0-based code layers."""
    if layer_index_base not in (0, 1):
        raise ValueError(f"layer_index_base must be 0 or 1, got {layer_index_base}")
    if layer_index_base == 0:
        return tuple(int(x) for x in layers)
    return tuple(int(x) - 1 for x in layers)


def default_table_sizes(max_ngram: int, *, base: int) -> tuple[int, ...]:
    if max_ngram < 2:
        raise ValueError("max_ngram must be >= 2")
    return tuple(int(base) for _ in range(2, max_ngram + 1))

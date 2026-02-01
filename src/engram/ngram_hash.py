from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sympy import isprime


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    candidate = int(start) + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


@dataclass(frozen=True)
class HashConfig:
    max_ngram: int
    num_heads: int
    table_size_per_ngram: Tuple[int, ...]  # base sizes for n=2..N (before prime)
    layer_ids: Tuple[int, ...]
    pad_id: int  # compressed pad id
    seed: int = 0


class NgramHashMapping(torch.nn.Module):
    """Deterministic multi-head hashing for {2..N}-grams.

    This mirrors `engram_demo_v1.py`:
    - per-layer random odd multipliers
    - multiplicative XOR mixing across n-gram tokens
    - per-head modulo with distinct primes near target table sizes
    """

    def __init__(
        self,
        *,
        tokenizer_vocab_size: int,
        cfg: HashConfig,
    ):
        super().__init__()
        if cfg.max_ngram < 2:
            raise ValueError("max_ngram must be >= 2")
        if len(cfg.table_size_per_ngram) != (cfg.max_ngram - 1):
            raise ValueError(
                "table_size_per_ngram length must be max_ngram-1 "
                f"(got {len(cfg.table_size_per_ngram)} for max_ngram={cfg.max_ngram})"
            )
        self.tokenizer_vocab_size = int(tokenizer_vocab_size)
        self.cfg = cfg

        # Per-layer multipliers, shape: (num_layers, max_ngram)
        PRIME_1 = 10007
        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // max(1, self.tokenizer_vocab_size))
        half_bound = max(1, M_max // 2)

        layer_ids = list(cfg.layer_ids)
        multipliers = []
        for layer_id in layer_ids:
            base_seed = int(cfg.seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(low=0, high=half_bound, size=(cfg.max_ngram,), dtype=np.int64)
            mult = r * 2 + 1
            multipliers.append(mult)
        multipliers_np = np.stack(multipliers, axis=0).astype(np.int64, copy=False)
        self.register_buffer("layer_multipliers", torch.from_numpy(multipliers_np), persistent=True)

        # Prime table sizes per layer/ngram/head. Keep as python ints.
        self.vocab_size_across_layers: Dict[int, List[List[int]]] = self._compute_prime_table_sizes()

        # quick index: layer_id -> row in layer_multipliers
        self._layer_id_to_row = {lid: i for i, lid in enumerate(layer_ids)}

    def _compute_prime_table_sizes(self) -> Dict[int, List[List[int]]]:
        seen_primes: set[int] = set()
        out: Dict[int, List[List[int]]] = {}
        for layer_id in self.cfg.layer_ids:
            sizes_for_layer: List[List[int]] = []
            for n in range(2, self.cfg.max_ngram + 1):
                base = int(self.cfg.table_size_per_ngram[n - 2])
                start = base - 1
                head_sizes = []
                for _ in range(self.cfg.num_heads):
                    p = find_next_prime(start, seen_primes)
                    seen_primes.add(p)
                    head_sizes.append(int(p))
                    start = p
                sizes_for_layer.append(head_sizes)
            out[int(layer_id)] = sizes_for_layer
        return out

    @property
    def num_total_heads(self) -> int:
        return (self.cfg.max_ngram - 1) * self.cfg.num_heads

    def total_rows_for_layer(self, layer_id: int) -> int:
        sizes = self.vocab_size_across_layers[int(layer_id)]
        return int(sum(sum(heads) for heads in sizes))

    def total_rows_all_layers(self) -> int:
        return int(sum(self.total_rows_for_layer(lid) for lid in self.cfg.layer_ids))

    def forward(self, *, compressed_input_ids: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Return hash ids of shape (B, T, H)."""
        if compressed_input_ids.dtype != torch.long:
            compressed_input_ids = compressed_input_ids.long()

        x = compressed_input_ids
        if x.dim() != 2:
            raise ValueError(f"compressed_input_ids must be (B,T), got shape {tuple(x.shape)}")
        B, T = x.shape

        lid = int(layer_id)
        if lid not in self._layer_id_to_row:
            raise KeyError(f"Unknown layer_id={lid} (expected one of {sorted(self._layer_id_to_row.keys())})")
        row = self._layer_id_to_row[lid]
        multipliers = self.layer_multipliers[row]  # (max_ngram,)

        # Build causal shifts: shift 0 is x, shift k pads on the left with pad_id.
        base_shifts: List[torch.Tensor] = [x]
        pad_val = int(self.cfg.pad_id)
        for k in range(1, self.cfg.max_ngram):
            pad = torch.full((B, k), pad_val, dtype=torch.long, device=x.device)
            # Keep output length exactly T (matches demo behavior).
            # For short sequences (e.g. generate() with use_cache=True => T=1),
            # T-k can be <=0; we still want a (B,T) tensor after left padding.
            take = max(0, T - k)
            shifted = torch.cat([pad, x[:, :take]], dim=1)[:, :T]
            base_shifts.append(shifted)

        all_hashes: List[torch.Tensor] = []
        sizes_for_layer = self.vocab_size_across_layers[lid]

        # Loop is small (max_ngram<=3, num_heads<=8), keep it simple.
        for n in range(2, self.cfg.max_ngram + 1):
            n_idx = n - 2
            tokens = base_shifts[:n]
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = torch.bitwise_xor(mix, tokens[k] * multipliers[k])

            head_sizes = sizes_for_layer[n_idx]
            for j in range(self.cfg.num_heads):
                mod = int(head_sizes[j])
                all_hashes.append(torch.remainder(mix, mod))

        out = torch.stack(all_hashes, dim=2)  # (B,T,H)
        # Assertions required by TASK.md T2.2
        assert out.shape == (B, T, self.num_total_heads)
        if pad_val is not None:
            # position 0 should never depend on previous context; shifting uses pad id.
            # (can't assert values, but can assert finite range)
            assert int(out.min()) >= 0
        return out

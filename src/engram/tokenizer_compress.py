from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from tokenizers import Regex, normalizers


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _default_demo_normalizer() -> normalizers.Normalizer:
    # Mirror engram_demo_v1.py
    SENTINEL = "\uE000"
    return normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ]
    )


@dataclass(frozen=True)
class CompressionStats:
    old_vocab_size: int
    new_vocab_size: int

    @property
    def reduction_ratio(self) -> float:
        if self.old_vocab_size <= 0:
            return 0.0
        return 1.0 - (self.new_vocab_size / self.old_vocab_size)


class TokenizerCompressor:
    """Build a surjective mapping old_token_id -> canonical_id.

    Two modes:
    - demo: matches `engram_demo_v1.py` normalizer rules + fallback handling.
    - paper: a simpler NFKC+lowercase+whitespace normalization (kept minimal).
    """

    def __init__(
        self,
        *,
        lookup_table: np.ndarray,
        num_new_tokens: int,
    ):
        if lookup_table.dtype != np.int64:
            lookup_table = lookup_table.astype(np.int64, copy=False)
        self.lookup_table_np = lookup_table
        self.num_new_tokens = int(num_new_tokens)

        # Lazily materialized torch buffer.
        self._lookup_table_torch: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return self.num_new_tokens

    def to_torch(self, device: torch.device) -> torch.Tensor:
        if self._lookup_table_torch is None or self._lookup_table_torch.device != device:
            self._lookup_table_torch = torch.from_numpy(self.lookup_table_np).to(device=device)
        return self._lookup_table_torch

    def compress(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compress raw input_ids to canonical ids. Supports arbitrary shapes."""
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        # Typical token ids are non-negative; keep the negative-safe path for robustness.
        if torch.any(input_ids < 0):
            lookup = self.to_torch(input_ids.device)
            out = input_ids.clone()
            mask = input_ids >= 0
            out[mask] = lookup[input_ids[mask]]
            return out

        lookup = self.to_torch(input_ids.device)
        return lookup[input_ids]

    @staticmethod
    def _build_lookup_table_demo(tokenizer) -> Tuple[np.ndarray, int]:
        normalizer = _default_demo_normalizer()

        old2new = {}
        key2new = {}
        new_tokens = []

        vocab_size = len(tokenizer)
        for tid in range(vocab_size):
            text = tokenizer.decode([tid], skip_special_tokens=False)
            if "�" in text:
                key = tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]
        return lookup, len(new_tokens)

    @staticmethod
    def _build_lookup_table_paper(tokenizer) -> Tuple[np.ndarray, int]:
        # Minimal normalization inferred from paper description (keep conservative).
        normalizer = normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Strip(),
            ]
        )

        old2new = {}
        key2new = {}
        new_tokens = []

        vocab_size = len(tokenizer)
        for tid in range(vocab_size):
            text = tokenizer.decode([tid], skip_special_tokens=False)
            norm = normalizer.normalize_str(text)
            key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]
        return lookup, len(new_tokens)

    @classmethod
    def build_or_load(
        cls,
        *,
        tokenizer,
        mode: str,
        cache_dir: str,
        tokenizer_fingerprint: Optional[str] = None,
    ) -> Tuple["TokenizerCompressor", CompressionStats]:
        mode = str(mode)
        if mode not in ("demo", "paper"):
            raise ValueError(f"tokenizer compression mode must be 'demo' or 'paper', got {mode!r}")

        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)

        # fingerprint should change when tokenizer vocab changes.
        fp = tokenizer_fingerprint or getattr(tokenizer, "name_or_path", "") or tokenizer.__class__.__name__
        cache_key = _sha1(f"{mode}::{fp}::{len(tokenizer)}")
        path = cache_root / f"lookup_{cache_key}.npz"

        if path.exists():
            data = np.load(str(path))
            lookup = data["lookup"]
            num_new = int(data["num_new"])
            compressor = cls(lookup_table=lookup, num_new_tokens=num_new)
            stats = CompressionStats(old_vocab_size=len(tokenizer), new_vocab_size=num_new)
            return compressor, stats

        if mode == "demo":
            lookup, num_new = cls._build_lookup_table_demo(tokenizer)
        else:
            lookup, num_new = cls._build_lookup_table_paper(tokenizer)

        # np.savez_compressed will append ".npz" if the filename doesn't end with it.
        tmp = str(path) + ".tmp.npz"
        np.savez_compressed(tmp, lookup=lookup, num_new=np.int64(num_new))
        os.replace(tmp, path)

        compressor = cls(lookup_table=lookup, num_new_tokens=num_new)
        stats = CompressionStats(old_vocab_size=len(tokenizer), new_vocab_size=num_new)
        return compressor, stats

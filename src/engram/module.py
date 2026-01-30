from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.engram.ngram_hash import HashConfig, NgramHashMapping
from src.engram.tokenizer_compress import CompressionStats, TokenizerCompressor


class MultiHeadEmbedding(nn.Module):
    """Single embedding table + offsets for multiple heads (demo-style)."""

    def __init__(self, head_sizes: Sequence[int], d_head: int):
        super().__init__()
        head_sizes = [int(x) for x in head_sizes]
        if not head_sizes:
            raise ValueError("head_sizes must be non-empty")
        if d_head <= 0:
            raise ValueError("d_head must be > 0")

        offsets = [0]
        for n in head_sizes[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=True)

        total = sum(head_sizes)
        self.embedding = nn.Embedding(num_embeddings=total, embedding_dim=int(d_head))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B,T,H)
        shifted = input_ids + self.offsets  # broadcast over (H,)
        return self.embedding(shifted)  # (B,T,H,d_head)


class ShortDepthwiseCausalConv(nn.Module):
    def __init__(self, hidden_size: int, *, kernel_size: int, dilation: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=int(kernel_size),
            groups=hidden_size,
            bias=False,
            padding=(int(kernel_size) - 1) * int(dilation),
            dilation=int(dilation),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        B, T, D = x.shape
        y = self.norm(x).transpose(1, 2)  # (B,D,T)
        y = self.conv(y)[..., :T]  # causal slice
        y = self.act(y).transpose(1, 2)  # (B,T,D)
        return y


@dataclass(frozen=True)
class EngramRuntimeInfo:
    old_vocab_size: int
    new_vocab_size: int
    num_heads_total: int
    d_head: int
    d_mem: int
    total_rows: int
    embed_params: int


class EngramLayer(nn.Module):
    """Engram module for a single transformer layer (single-stream backbone)."""

    def __init__(
        self,
        *,
        layer_id: int,
        tokenizer,
        hidden_size: int,
        max_ngram: int,
        num_heads: int,
        table_size_per_ngram: Sequence[int],
        pad_token_id: int,
        seed: int,
        kernel_size: int,
        d_mem: Optional[int],
        tokenizer_compression_mode: str,
        gating_mode: str,
        init_equivalence: str,
        cache_dir: str,
    ):
        super().__init__()
        self.layer_id = int(layer_id)
        self.hidden_size = int(hidden_size)
        self.max_ngram = int(max_ngram)
        self.num_heads = int(num_heads)
        self.kernel_size = int(kernel_size)
        self.d_mem = int(d_mem) if d_mem is not None else (self.hidden_size // 2)
        self.gating_mode = str(gating_mode)
        self.init_equivalence = str(init_equivalence)

        if self.gating_mode not in ("paper", "demo"):
            raise ValueError(f"gating_mode must be 'paper' or 'demo', got {self.gating_mode!r}")

        # Tokenizer compression (cached on disk).
        compressor, stats = TokenizerCompressor.build_or_load(
            tokenizer=tokenizer,
            mode=str(tokenizer_compression_mode),
            cache_dir=cache_dir,
        )
        self.compression_stats = stats
        self.compressor = compressor

        # Compressed pad id.
        raw_pad = int(pad_token_id)
        pad_comp = int(compressor.lookup_table_np[raw_pad])
        self.pad_id_compressed = pad_comp

        # Hash mapping.
        hcfg = HashConfig(
            max_ngram=self.max_ngram,
            num_heads=self.num_heads,
            table_size_per_ngram=tuple(int(x) for x in table_size_per_ngram),
            layer_ids=(self.layer_id,),
            pad_id=self.pad_id_compressed,
            seed=int(seed),
        )
        self.hash_mapping = NgramHashMapping(tokenizer_vocab_size=len(compressor), cfg=hcfg)

        # Embedding sizes.
        heads_total = self.hash_mapping.num_total_heads
        if self.d_mem % heads_total != 0:
            raise ValueError(f"d_mem must be divisible by total heads={heads_total}, got d_mem={self.d_mem}")
        d_head = self.d_mem // heads_total

        # Prime sizes per head for this layer.
        head_sizes = [x for group in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in group]
        self.memory_embedding = MultiHeadEmbedding(head_sizes=head_sizes, d_head=d_head)

        self.key_proj = nn.Linear(self.d_mem, self.hidden_size)
        self.value_proj = nn.Linear(self.d_mem, self.hidden_size)
        self.q_norm = nn.RMSNorm(self.hidden_size)
        self.k_norm = nn.RMSNorm(self.hidden_size)

        self.short_conv = ShortDepthwiseCausalConv(
            self.hidden_size, kernel_size=self.kernel_size, dilation=self.max_ngram
        )

        self.reset_parameters()

        total_rows = self.hash_mapping.total_rows_for_layer(self.layer_id)
        embed_params = int(total_rows) * int(d_head)
        self.runtime_info = EngramRuntimeInfo(
            old_vocab_size=stats.old_vocab_size,
            new_vocab_size=stats.new_vocab_size,
            num_heads_total=heads_total,
            d_head=d_head,
            d_mem=self.d_mem,
            total_rows=total_rows,
            embed_params=embed_params,
        )

    def reset_parameters(self) -> None:
        # Keep default init for most layers.
        # D7: allow a strict "zero_output" init so baseline forward is preserved.
        if self.init_equivalence == "zero_output":
            nn.init.zeros_(self.value_proj.weight)
            if self.value_proj.bias is not None:
                nn.init.zeros_(self.value_proj.bias)
            nn.init.zeros_(self.short_conv.conv.weight)
        elif self.init_equivalence == "none":
            pass
        else:
            raise ValueError(f"init_equivalence must be 'zero_output' or 'none', got {self.init_equivalence!r}")

    def forward(self, *, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B,T,D)
        # input_ids: (B,T)
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be (B,T,D), got {tuple(hidden_states.shape)}")
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be (B,T), got {tuple(input_ids.shape)}")

        # Hashing operates on compressed ids.
        compressed = self.compressor.compress(input_ids)
        hash_ids = self.hash_mapping(compressed_input_ids=compressed, layer_id=self.layer_id)  # (B,T,H)

        # Retrieve embeddings and flatten.
        e = self.memory_embedding(hash_ids).flatten(start_dim=2)  # (B,T,d_mem)

        # Key/Value projections.
        k = self.key_proj(e)
        v = self.value_proj(e)

        # Gate (scalar per token).
        qn = self.q_norm(hidden_states)
        kn = self.k_norm(k)
        # Compute gate logits in fp32 for stability, then cast back.
        dot = (qn.float() * kn.float()).sum(dim=-1) / math.sqrt(self.hidden_size)
        if self.gating_mode == "demo":
            # Mirror demo's extra transform before sigmoid.
            dot = dot.abs().clamp_min(1e-6).sqrt() * dot.sign()
        gate = torch.sigmoid(dot).to(dtype=hidden_states.dtype).unsqueeze(-1)  # (B,T,1)

        gated_v = gate * v
        y = self.short_conv(gated_v) + gated_v
        return y

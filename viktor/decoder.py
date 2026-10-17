"""Small Qwen3-style decoder-only LM: RMSNorm, GQA, QK-norm, RoPE, SwiGLU.

Two call modes share the same weights:
  * full causal pass over [signal prefix | tokens] (training and prefill; optionally writes the KV cache)
  * single-token step against a static KV cache, each sequence at its own position (decode)
This is the reference / training implementation; fast_decoder.py runs the same weights on
fused Triton kernels for inference.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import apply_rope, rope_tables

try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_func = flash_attn_with_kvcache = None


@dataclass
class DecoderConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 64
    ffn_dim: int = 1536
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    max_seq_len: int = 4096


def _use_flash(x):
    return flash_attn_func is not None and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16)


class KVCache:
    """Static cache, layout [layer, batch, position, kv_head, head_dim] (flash-attn kvcache layout)."""

    def __init__(self, cfg: DecoderConfig, batch: int, max_len: int, dtype, device):
        shape = (cfg.n_layers, batch, max_len, cfg.n_kv_heads, cfg.head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_len = max_len
        self.positions = torch.arange(max_len, device=device)


class Attention(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.h, self.hkv, self.d = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.qkv_proj = nn.Linear(cfg.d_model, (self.h + 2 * self.hkv) * self.d, bias=False)
        self.o_proj = nn.Linear(self.h * self.d, cfg.d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.d, eps=cfg.norm_eps)
        self.k_norm = nn.RMSNorm(self.d, eps=cfg.norm_eps)

    def forward(self, x, cos, sin, layer, cache: Optional[KVCache], batch_slice, pos, seqlens):
        B, T, _ = x.shape
        q, k, v = self.qkv_proj(x).view(B, T, self.h + 2 * self.hkv, self.d).split([self.h, self.hkv, self.hkv], 2)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        if pos is None:
            if cache is not None:
                cache.k[layer, batch_slice, :T] = k
                cache.v[layer, batch_slice, :T] = v
            o = self._causal(q, k, v)
        else:
            o = self._decode(q, k, v, layer, cache, pos, seqlens)
        return self.o_proj(o.reshape(B, T, self.h * self.d))

    def _causal(self, q, k, v):
        if _use_flash(q):
            return flash_attn_func(q, k, v, causal=True)
        rep = self.h // self.hkv
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2).repeat_interleave(rep, 1),
            v.transpose(1, 2).repeat_interleave(rep, 1), is_causal=True).transpose(1, 2)

    def _decode(self, q, k, v, layer, cache: KVCache, pos, seqlens):
        kc, vc = cache.k[layer], cache.v[layer]
        if _use_flash(q):
            # writes k/v at index seqlens in place and attends to positions [0, seqlens]
            return flash_attn_with_kvcache(q, kc, vc, k=k, v=v, cache_seqlens=seqlens, causal=True)
        rows = torch.arange(kc.shape[0], device=kc.device)
        kc[rows, pos] = k[:, 0]
        vc[rows, pos] = v[:, 0]
        rep = self.h // self.hkv
        mask = (cache.positions[None, :] <= pos[:, None])[:, None, None, :]
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), kc.transpose(1, 2).repeat_interleave(rep, 1),
            vc.transpose(1, 2).repeat_interleave(rep, 1), attn_mask=mask).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, layer, cache, batch_slice, pos, seqlens):
        x = x + self.attn(self.attn_norm(x), cos, sin, layer, cache, batch_slice, pos, seqlens)
        gate, up = self.gate_up(self.mlp_norm(x)).chunk(2, -1)
        return x + self.down(F.silu(gate) * up)


class Decoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, cache: Optional[KVCache] = None, batch_slice=slice(None), pos=None, seqlens=None):
        """x: [B, T, d] input embeddings. pos ([B], position of each sequence's token) selects
        single-token decode mode, with seqlens = pos as int32. Returns final-normed hidden states;
        apply `head` for logits."""
        if pos is None:
            cos, sin = self.cos[:x.shape[1]], self.sin[:x.shape[1]]
        else:
            cos, sin = self.cos[pos][:, None], self.sin[pos][:, None]
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, i, cache, batch_slice, pos, seqlens)
        return self.norm(x)

    def head(self, h):
        return self.lm_head(h)

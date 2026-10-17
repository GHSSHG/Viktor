"""Inference forward of the Decoder on the fused Triton kernels (same weights as decoder.Decoder).

Tokens are packed [n_tokens, d]: prefill packs the prefixes of many variable-length chunks
(flash_attn_varlen_func), a decode step is one token per sequence (flash_attn_with_kvcache).
The matmuls stay on cuBLAS.
"""

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from .decoder import Decoder, KVCache
from .triton_ops import add_rmsnorm, qkv_rope_cache, swiglu


class FastDecoder:
    def __init__(self, dec: Decoder, attention_backend="auto"):
        cfg = dec.cfg
        assert all(n & (n - 1) == 0 for n in (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim // 2)), \
            "kernels need power-of-two head counts and head_dim / 2"
        self.dec, self.cfg = dec, cfg
        if attention_backend == "auto":
            # decode_attention.py is tuned for sm80, bf16, GQA 8/2, head_dim 64; FlashAttention otherwise
            weight = dec.lm_head.weight
            tuned = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim) == (8, 2, 64)
            tuned = tuned and weight.dtype == torch.bfloat16 and weight.is_cuda
            tuned = tuned and torch.cuda.get_device_capability(weight.device) == (8, 0)
            attention_backend = "triton" if tuned else "flash"
        if attention_backend not in ("triton", "flash"):
            raise ValueError(f"unknown decode attention backend: {attention_backend}")
        self.attention_backend = attention_backend

    def _layers(self, x, pos, seq, cache: KVCache, attend, return_kv, last_indices=None):
        """x [n, d] residual stream (modified in place). Returns the final-normed hidden [n, d]."""
        cfg, layers = self.cfg, self.dec.layers
        h = add_rmsnorm(x, layers[0].attn_norm.weight, cfg.norm_eps)
        for i, blk in enumerate(layers):
            a = blk.attn
            qkv = F.linear(h, a.qkv_proj.weight)
            r = qkv_rope_cache(qkv, a.q_norm.weight, a.k_norm.weight, self.dec.cos, self.dec.sin, pos, seq,
                               cache.k[i], cache.v[i], cfg.norm_eps, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                               return_kv=return_kv)
            if last_indices is not None and i == len(layers) - 1:
                # The final signal outputs have no consumers. Keep all K/V, but
                # only finish the BOS residual stream and its attention queries.
                x = x[last_indices]
                q, k, v = r
                r = q[last_indices], k, v
            o = attend(i, r)
            h = add_rmsnorm(x, blk.mlp_norm.weight, cfg.norm_eps, delta=F.linear(o.reshape(x.shape[0], -1),
                                                                                    a.o_proj.weight))
            m = F.linear(swiglu(F.linear(h, blk.gate_up.weight)), blk.down.weight)
            w_next = layers[i + 1].attn_norm.weight if i + 1 < len(layers) else self.dec.norm.weight
            h = add_rmsnorm(x, w_next, cfg.norm_eps, delta=m)
        return h

    def prefill(self, x, pos, seq, cu_seqlens, max_seqlen, cache: KVCache, last_indices=None, cu_queries=None):
        """x [n, d] packed prefix embeddings of several sequences; pos/seq [n] int32; cu_seqlens [b+1] int32.
        Fills the cache rows of those sequences. Returns hidden [n, d], or [b, d] at the final
        position of each sequence when last_indices and cu_queries are provided together."""
        assert (last_indices is None) == (cu_queries is None)
        def attend(i, r):
            q, k, v = r
            if last_indices is not None and i == self.cfg.n_layers - 1:
                # BOS is the last position and may attend to the entire prefix.
                return flash_attn_varlen_func(q, k, v, cu_queries, cu_seqlens, 1, max_seqlen, causal=False)
            return flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True)
        return self._layers(x, pos, seq, cache, attend, return_kv=True, last_indices=last_indices)

    def decode(self, x, pos, seq, cache_seqlens, cache: KVCache):
        """One token per sequence: x [B, d], pos/seq [B] int32, cache_seqlens = pos + 1 (int32)."""
        def attend(i, q):
            if self.attention_backend == "triton":
                from .decode_attention import decode_attention
                return decode_attention(q, cache.k[i], cache.v[i], cache_seqlens)
            # The cache is already written, so FA2 would pick its general forward kernel;
            # identity cache_batch_idx with num_splits=1 selects the split-KV decode kernel.
            return flash_attn_with_kvcache(q[:, None], cache.k[i], cache.v[i], cache_seqlens=cache_seqlens,
                                          cache_batch_idx=seq, num_splits=1)[:, 0]
        return self._layers(x, pos, seq, cache, attend, return_kv=False)

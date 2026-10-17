"""Fused Triton kernels for the decoder, on the packed token layout [n_tokens, ...].

Every op is per token, so the same kernels serve variable-length prefill (tokens of many
chunks concatenated) and decode steps (one token per sequence):
  add_rmsnorm      x += delta (in place); h = RMSNorm(x) * w
  qkv_rope_cache   QK-norm + RoPE on q and k, write k/v into the KV cache at (sequence, position),
                   optionally also return packed k/v for the prefill attention
  swiglu           silu(gate) * up
Rounding follows the PyTorch reference (bf16 storage between ops, fp32 math inside an op).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm_kernel(x_ptr, d_ptr, w_ptr, h_ptr, n_cols, eps,
                        HAS_DELTA: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=m, other=0.0)
    if HAS_DELTA:
        x = (x.to(tl.float32) + tl.load(d_ptr + row * n_cols + cols, mask=m, other=0.0).to(tl.float32)).to(
            x_ptr.dtype.element_ty)
        tl.store(x_ptr + row * n_cols + cols, x, mask=m)
    xf = x.to(tl.float32)
    r = tl.rsqrt(tl.sum(xf * xf, 0) / n_cols + eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(h_ptr + row * n_cols + cols, (xf * r * w).to(h_ptr.dtype.element_ty), mask=m)


def add_rmsnorm(x, w, eps, delta=None):
    """x [n, d] (updated in place by += delta); returns RMSNorm(x) * w."""
    n, d = x.shape
    h = torch.empty_like(x)
    _add_rmsnorm_kernel[(n,)](x, x if delta is None else delta, w, h, d, eps,
                              HAS_DELTA=delta is not None, BLOCK=triton.next_power_of_2(d), num_warps=4)
    return h


@triton.jit
def _norm_rope(x1, x2, w_ptr, c, s, eps, D: tl.constexpr, HALF: tl.constexpr, OUT_TY: tl.constexpr):
    d = tl.arange(0, HALF)
    r = tl.rsqrt((tl.sum(x1 * x1, 1) + tl.sum(x2 * x2, 1)) / D + eps)
    y1 = (x1 * r[:, None] * tl.load(w_ptr + d).to(tl.float32)[None, :]).to(OUT_TY).to(tl.float32)
    y2 = (x2 * r[:, None] * tl.load(w_ptr + HALF + d).to(tl.float32)[None, :]).to(OUT_TY).to(tl.float32)
    return y1 * c[None, :] - y2 * s[None, :], y1 * s[None, :] + y2 * c[None, :]


@triton.jit
def _qkv_rope_cache_kernel(qkv_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, pos_ptr, seq_ptr,
                           q_ptr, kc_ptr, vc_ptr, ko_ptr, vo_ptr, stride_cb, stride_cs, eps,
                           H: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, WRITE_KV: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    HALF: tl.constexpr = D // 2
    OUT_TY: tl.constexpr = q_ptr.dtype.element_ty
    d = tl.arange(0, HALF)
    pos = tl.load(pos_ptr + t).to(tl.int64)
    seq = tl.load(seq_ptr + t).to(tl.int64)
    c = tl.load(cos_ptr + pos * HALF + d)
    s = tl.load(sin_ptr + pos * HALF + d)
    row = qkv_ptr + t * (H + 2 * HKV) * D

    hq = tl.arange(0, H)
    q1 = tl.load(row + hq[:, None] * D + d[None, :]).to(tl.float32)
    q2 = tl.load(row + hq[:, None] * D + HALF + d[None, :]).to(tl.float32)
    o1, o2 = _norm_rope(q1, q2, qw_ptr, c, s, eps, D, HALF, OUT_TY)
    qo = q_ptr + t * H * D + hq[:, None] * D + d[None, :]
    tl.store(qo, o1.to(OUT_TY))
    tl.store(qo + HALF, o2.to(OUT_TY))

    hk = tl.arange(0, HKV)
    k1 = tl.load(row + (H + hk[:, None]) * D + d[None, :]).to(tl.float32)
    k2 = tl.load(row + (H + hk[:, None]) * D + HALF + d[None, :]).to(tl.float32)
    o1, o2 = _norm_rope(k1, k2, kw_ptr, c, s, eps, D, HALF, OUT_TY)
    v1 = tl.load(row + (H + HKV + hk[:, None]) * D + d[None, :])
    v2 = tl.load(row + (H + HKV + hk[:, None]) * D + HALF + d[None, :])
    off = seq * stride_cb + pos * stride_cs + hk[:, None] * D + d[None, :]
    tl.store(kc_ptr + off, o1.to(OUT_TY))
    tl.store(kc_ptr + off + HALF, o2.to(OUT_TY))
    tl.store(vc_ptr + off, v1)
    tl.store(vc_ptr + off + HALF, v2)
    if WRITE_KV:
        po = t * HKV * D + hk[:, None] * D + d[None, :]
        tl.store(ko_ptr + po, o1.to(OUT_TY))
        tl.store(ko_ptr + po + HALF, o2.to(OUT_TY))
        tl.store(vo_ptr + po, v1)
        tl.store(vo_ptr + po + HALF, v2)


def qkv_rope_cache(qkv, q_norm_w, k_norm_w, cos, sin, pos, seq, k_cache, v_cache, eps, n_heads, n_kv_heads,
                   head_dim, return_kv=False):
    """qkv [n, (H + 2 Hkv) * D]; pos/seq [n] position and cache row of each token; cos/sin [S, D/2] fp32;
    k_cache/v_cache [B, S, Hkv, D] (one layer). Returns q [n, H, D] (and packed k, v [n, Hkv, D])."""
    n = qkv.shape[0]
    q = torch.empty(n, n_heads, head_dim, dtype=qkv.dtype, device=qkv.device)
    k = v = q
    if return_kv:
        k = torch.empty(n, n_kv_heads, head_dim, dtype=qkv.dtype, device=qkv.device)
        v = torch.empty_like(k)
    _qkv_rope_cache_kernel[(n,)](qkv, q_norm_w, k_norm_w, cos, sin, pos, seq, q, k_cache, v_cache, k, v,
                                 k_cache.stride(0), k_cache.stride(1), eps, H=n_heads, HKV=n_kv_heads,
                                 D=head_dim, WRITE_KV=return_kv, num_warps=2)
    return (q, k, v) if return_kv else q


@triton.jit
def _swiglu_kernel(gu_ptr, out_ptr, F, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < F
    g = tl.load(gu_ptr + row * 2 * F + cols, mask=m, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * 2 * F + F + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * F + cols, (g * tl.sigmoid(g) * u).to(out_ptr.dtype.element_ty), mask=m)


def swiglu(gu):
    """gu [n, 2F] (gate | up) -> silu(gate) * up [n, F]."""
    n, f2 = gu.shape
    out = torch.empty(n, f2 // 2, dtype=gu.dtype, device=gu.device)
    BLOCK = 512
    _swiglu_kernel[(n, triton.cdiv(f2 // 2, BLOCK))](gu, out, f2 // 2, BLOCK=BLOCK, num_warps=4)
    return out

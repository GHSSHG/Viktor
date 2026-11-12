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
def _qkv_rope_cache_kernel(qkv_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, pos_ptr, rpos_ptr, seq_ptr,
                           q_ptr, kc_ptr, vc_ptr, ko_ptr, vo_ptr, stride_cb, stride_cs, eps,
                           H: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, WRITE_KV: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    HALF: tl.constexpr = D // 2
    OUT_TY: tl.constexpr = q_ptr.dtype.element_ty
    d = tl.arange(0, HALF)
    pos = tl.load(pos_ptr + t).to(tl.int64)
    seq = tl.load(seq_ptr + t).to(tl.int64)
    rpos = tl.load(rpos_ptr + t).to(tl.int64)
    c = tl.load(cos_ptr + rpos * HALF + d)
    s = tl.load(sin_ptr + rpos * HALF + d)
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
                   head_dim, return_kv=False, rope_pos=None):
    """qkv [n, (H + 2 Hkv) * D]; pos/seq [n] position and cache row of each token; cos/sin [S, D/2] fp32;
    k_cache/v_cache [B, S, Hkv, D] (one layer). rope_pos [n]: RoPE positions when they differ from pos.
    Returns q [n, H, D] (and packed k, v [n, Hkv, D])."""
    n = qkv.shape[0]
    q = torch.empty(n, n_heads, head_dim, dtype=qkv.dtype, device=qkv.device)
    k = v = q
    if return_kv:
        k = torch.empty(n, n_kv_heads, head_dim, dtype=qkv.dtype, device=qkv.device)
        v = torch.empty_like(k)
    _qkv_rope_cache_kernel[(n,)](qkv, q_norm_w, k_norm_w, cos, sin, pos, pos if rope_pos is None else rope_pos, seq, q, k_cache, v_cache, k, v,
                                 k_cache.stride(0), k_cache.stride(1), eps, H=n_heads, HKV=n_kv_heads,
                                 D=head_dim, WRITE_KV=return_kv, num_warps=2)
    return (q, k, v) if return_kv else q


_LINEAR_CONFIGS = [triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=w, num_stages=s)
                   for bm, bn, bk, w, s in [(16, 32, 128, 4, 3), (16, 32, 128, 4, 4), (32, 32, 128, 4, 4),
                                            (32, 64, 64, 4, 3), (32, 64, 64, 4, 4), (32, 64, 64, 2, 3),
                                            (64, 64, 128, 4, 3), (64, 128, 64, 4, 3)]]


@triton.autotune(configs=_LINEAR_CONFIGS, key=["M", "N", "K", "SWIGLU"])
@triton.jit
def _linear_kernel(x_ptr, w_ptr, y_ptr, M, N, K, SWIGLU: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    acc2 = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(x_ptr + rm[:, None] * K + (k0 + rk)[None, :], mask=rm[:, None] < M, other=0.0)
        acc += tl.dot(x, tl.load(w_ptr + rn[None, :] * K + (k0 + rk)[:, None], mask=rn[None, :] < N, other=0.0))
        if SWIGLU:
            acc2 += tl.dot(x, tl.load(w_ptr + (rn[None, :] + N) * K + (k0 + rk)[:, None]))
    out_ty = y_ptr.dtype.element_ty
    if SWIGLU:  # rounded as the unfused path: gate / up and silu(gate) stored in bf16
        g = acc.to(out_ty).to(tl.float32)
        acc = (g * tl.sigmoid(g)).to(out_ty).to(tl.float32) * acc2.to(out_ty).to(tl.float32)
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc.to(out_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


def linear(x, w, swiglu=False, out_dtype=None):
    """Small-M x @ w^T (decode steps: one row per sequence); faster than cuBLAS at M <= 256. swiglu: w is
    gate_up [2F, K] and the result silu(gate) * up [M, F] (F a multiple of 128). K a multiple of 128.
    out_dtype: e.g. fp32 logits (not rounded to bf16 before the argmax)."""
    M, K = x.shape
    N = w.shape[0] // 2 if swiglu else w.shape[0]
    y = torch.empty(M, N, dtype=out_dtype or x.dtype, device=x.device)
    _linear_kernel[lambda m: (triton.cdiv(M, m["BM"]), triton.cdiv(N, m["BN"]))](x, w, y, M, N, K, SWIGLU=swiglu)
    return y


@triton.jit
def _embed_rmsnorm_kernel(tok_ptr, emb_ptr, w_ptr, x_ptr, h_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    t = tl.load(tok_ptr + row).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < n_cols
    x = tl.load(emb_ptr + t * n_cols + cols, mask=m, other=0.0)
    tl.store(x_ptr + row * n_cols + cols, x, mask=m)
    xf = x.to(tl.float32)
    r = tl.rsqrt(tl.sum(xf * xf, 0) / n_cols + eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(h_ptr + row * n_cols + cols, (xf * r * w).to(h_ptr.dtype.element_ty), mask=m)


def embed_rmsnorm(tok, emb, w, eps):
    """tok [n] token ids -> (x = emb[tok] [n, d], RMSNorm(x) * w): the embedding lookup and the first norm."""
    n, d = tok.shape[0], emb.shape[1]
    x = torch.empty(n, d, dtype=emb.dtype, device=emb.device)
    h = torch.empty_like(x)
    _embed_rmsnorm_kernel[(n,)](tok, emb, w, x, h, d, eps, BLOCK=triton.next_power_of_2(d), num_warps=4)
    return x, h


@triton.jit
def _greedy_update_kernel(logits_ptr, bias_ptr, tail_ptr, homo_ptr, tok_ptr, run_ptr, pos_ptr, plen_ptr, roff_ptr,
                          out_ptr, pos32_ptr, seqlens_ptr, rpos32_ptr, V, out_stride, homo_stop,
                          EOS: tl.constexpr, PAD: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < V
    z = tl.load(logits_ptr + b * V + cols, mask=m, other=float("-inf")).to(tl.float32)
    z = z + tl.load(bias_ptr + cols, mask=m, other=0.0)
    nxt = tl.argmax(z, 0, tie_break_left=True).to(tl.int64)
    prev = tl.load(tok_ptr + b)
    nxt = tl.where(tl.load(tail_ptr + prev) != 0, EOS, nxt)  # a tail must be last
    if homo_stop > 0:
        same = (nxt == prev) & (tl.load(homo_ptr + nxt) != 0)
        run = tl.where(same, tl.load(run_ptr + b) + 1, 0)
        tl.store(run_ptr + b, run)
        nxt = tl.where(run + 1 >= homo_stop, EOS, nxt)
    nxt = tl.where((prev == EOS) | (prev == PAD), PAD, nxt)
    pos = tl.load(pos_ptr + b)
    tl.store(out_ptr + b * out_stride + pos - tl.load(plen_ptr + b), nxt)
    tl.store(tok_ptr + b, nxt)
    pos += 1
    tl.store(pos_ptr + b, pos)
    tl.store(pos32_ptr + b, pos.to(tl.int32))
    tl.store(seqlens_ptr + b, (pos + 1).to(tl.int32))
    tl.store(rpos32_ptr + b, (pos + tl.load(roff_ptr + b)).to(tl.int32))


def greedy_update(logits, bias, is_tail, is_homo, homo_stop, tok, run, pos, plen, rope_off, out, pos32, seqlens,
                  rpos32, eos, pad):
    """One greedy decode step's bookkeeping in one kernel (DecodeEngine._select + output / state updates):
    token = argmax(logits + bias); EOS after a tail token or when a homopolymer run reaches homo_stop tokens; PAD
    after EOS / PAD. Writes it to out[b, pos - plen], tok, and advances pos (+ pos32, seqlens = pos + 1,
    rpos32 = pos + rope_off for the next step)."""
    B, V = logits.shape
    _greedy_update_kernel[(B,)](logits, bias, is_tail, is_homo, tok, run, pos, plen, rope_off, out, pos32, seqlens,
                                rpos32, V, out.stride(0), homo_stop, EOS=eos, PAD=pad,
                                BLOCK=triton.next_power_of_2(V), num_warps=4)


@triton.jit
def _swiglu_kernel(gu_ptr, out_ptr, F, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < F
    g = tl.load(gu_ptr + row * 2 * F + cols, mask=m, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * 2 * F + F + cols, mask=m, other=0.0).to(tl.float32)
    act = (g * tl.sigmoid(g)).to(out_ptr.dtype.element_ty).to(tl.float32)  # F.silu output is stored, as in torch
    tl.store(out_ptr + row * F + cols, (act * u).to(out_ptr.dtype.element_ty), mask=m)


def swiglu(gu):
    """gu [n, 2F] (gate | up) -> silu(gate) * up [n, F]."""
    n, f2 = gu.shape
    out = torch.empty(n, f2 // 2, dtype=gu.dtype, device=gu.device)
    BLOCK = 512
    _swiglu_kernel[(n, triton.cdiv(f2 // 2, BLOCK))](gu, out, f2 // 2, BLOCK=BLOCK, num_warps=4)
    return out

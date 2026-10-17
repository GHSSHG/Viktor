"""Single-query GQA attention on the existing contiguous BF16 KV cache.

One CTA reuses each KV tile across its query head group. FP32 online softmax;
optional sequence splits are combined using log-sum-exp, never by averaging.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _decode(qp, kp, vp, lp, op, lsep, stride_kb, stride_ks,
            H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
            BN: tl.constexpr, SPLITS: tl.constexpr):
    b, hk, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    g = tl.arange(0, 16)
    d = tl.arange(0, D)
    n = tl.arange(0, BN)
    q = tl.load(qp + b * H * D + (hk * G + g[:, None]) * D + d[None, :],
                mask=g[:, None] < G, other=0)
    length = tl.load(lp + b)
    blocks = tl.cdiv(length, BN * SPLITS)
    begin = part * blocks * BN
    end = tl.minimum(begin + blocks * BN, length)
    m = tl.full((16,), float('-inf'), tl.float32)
    denom = tl.full((16,), 0, tl.float32)
    acc = tl.full((16, D), 0, tl.float32)
    for start in range(begin, end, BN):
        p = start + n
        k = tl.load(kp + b * stride_kb + p[None, :] * stride_ks + hk * D + d[:, None],
                    mask=p[None, :] < length, other=0)
        scores = tl.dot(q, k) * (D ** -0.5 * 1.4426950408889634)
        scores = tl.where(p[None, :] < length, scores, float('-inf'))
        mx = tl.maximum(m, tl.max(scores, 1))
        alpha = tl.exp2(m - mx)
        probs = tl.exp2(scores - mx[:, None])
        denom = denom * alpha + tl.sum(probs, 1)
        v = tl.load(vp + b * stride_kb + p[:, None] * stride_ks + hk * D + d[None, :],
                    mask=p[:, None] < length, other=0)
        acc = acc * alpha[:, None] + tl.dot(probs.to(v.dtype), v)
        m = mx
    out = acc / tl.where(denom > 0, denom, 1)[:, None]
    if SPLITS == 1:
        off = b * H * D + (hk * G + g[:, None]) * D + d[None, :]
    else:
        off = ((b * HK + hk) * SPLITS + part) * G * D + g[:, None] * D + d[None, :]
        loff = ((b * HK + hk) * SPLITS + part) * G + g
        tl.store(lsep + loff, tl.where(denom > 0, m + tl.log2(denom), float('-inf')), mask=g < G)
    tl.store(op + off, out, mask=g[:, None] < G)


@triton.jit
def _combine(pp, lp, op, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
             G: tl.constexpr, SPLITS: tl.constexpr):
    b, h = tl.program_id(0), tl.program_id(1)
    s, d = tl.arange(0, SPLITS), tl.arange(0, D)
    base = (b * HK + h // G) * SPLITS * G + h % G
    lse = tl.load(lp + base + s * G)
    w = tl.exp2(lse - tl.max(lse, 0))
    w = w / tl.sum(w, 0)
    p = tl.load(pp + (base + s[:, None] * G) * D + d[None, :])
    out = tl.sum(p * w[:, None], 0)
    tl.store(op + (b * H + h) * D + d, out)


def decode_attention(q, k, v, lengths, block=64, splits=1, warps=4):
    """q [B,H,D], KV [B,S,Hkv,D], lengths [B] int32 (includes current token)."""
    B, H, D = q.shape
    HK = k.shape[2]
    assert q.is_contiguous() and H % HK == 0 and H // HK <= 16 and D == 64
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert k.stride() == v.stride() and k.stride(-1) == 1 and k.stride(-2) == D
    assert splits in (1, 2, 4, 8)
    out = torch.empty_like(q)
    partial = out if splits == 1 else torch.empty((B, HK, splits, H // HK, D), device=q.device, dtype=torch.float32)
    lse = out if splits == 1 else torch.empty((B, HK, splits, H // HK), device=q.device, dtype=torch.float32)
    _decode[(B, HK, splits)](q, k, v, lengths, partial, lse, k.stride(0), k.stride(1),
                            H, HK, D, H // HK, block, splits, num_warps=warps, num_stages=2)
    if splits > 1:
        _combine[(B, H)](partial, lse, out, H, HK, D, H // HK, splits, num_warps=4)
    return out

"""Single-query GQA decode attention on the contiguous BF16 KV cache [B, S, Hkv, D].

One program per (sequence, KV head): each K/V tile is loaded once and serves the whole query-head
group, with an FP32 online softmax. Tuned for sm80, GQA 8/2, head_dim 64 (FastDecoder "auto").
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _decode(qp, kp, vp, lp, op, stride_kb, stride_ks,
            H: tl.constexpr, D: tl.constexpr, G: tl.constexpr, BN: tl.constexpr):
    b, hk = tl.program_id(0), tl.program_id(1)
    g = tl.arange(0, 16)  # query heads of the group, padded to the minimum tl.dot size
    d = tl.arange(0, D)
    n = tl.arange(0, BN)
    q = tl.load(qp + b * H * D + (hk * G + g[:, None]) * D + d[None, :], mask=g[:, None] < G, other=0)
    length = tl.load(lp + b)
    m = tl.full((16,), float('-inf'), tl.float32)
    denom = tl.full((16,), 0, tl.float32)
    acc = tl.full((16, D), 0, tl.float32)
    for start in range(0, length, BN):
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
    tl.store(op + b * H * D + (hk * G + g[:, None]) * D + d[None, :], out, mask=g[:, None] < G)


def decode_attention(q, k, v, lengths, block=64):
    """q [B, H, D]; k/v [B, S, Hkv, D] (one layer of the cache); lengths [B] int32, including the
    current token, 1 <= lengths <= S. Only the valid prefix of each row is read."""
    B, H, D = q.shape
    HK = k.shape[2]
    assert q.is_contiguous() and H % HK == 0 and H // HK <= 16 and D == 64
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert k.stride() == v.stride() and k.stride(-1) == 1 and k.stride(-2) == D
    assert lengths.shape == (B,) and lengths.dtype == torch.int32 and lengths.is_contiguous()
    out = torch.empty_like(q)
    _decode[(B, HK)](q, k, v, lengths, out, k.stride(0), k.stride(1), H, D, H // HK, block,
                     num_warps=4, num_stages=2)
    return out

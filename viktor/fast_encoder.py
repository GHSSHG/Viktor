"""FAST encoder (trained from scratch): conv stack to 12x, then transformer stages at 12x / 24x / 48x.

  conv  1 -> 16 (k5 s1) -> 16 (k5 s1) -> 32 (k9 s3) -> 64 (k9 s2) -> 128 (k5 s2), SiLU   12x, 1024 frames / 12288 samples
  12x   3 layers, d128, 2 heads, window +-64 frames
  24x   RMSNorm + Conv k3 s2 128 -> 256, 2 layers, d256, 4 heads, window +-32
  48x   RMSNorm + Conv k3 s2 256 -> 512, 2 layers, d512, 8 heads, window +-32          48x, 256 frames, d_model 512
Layers: pre-RMSNorm, QK-norm, RoPE (positions within the chunk), sliding-window attention, SwiGLU.
Variable lengths in one batch (lens: real samples per row, multiples of the stride): everything past a row's length
is zeroed after every conv and stage, so each row is encoded as if zero-padded at its own end (as a chunk of that
length alone), and attention never reads past it.
CTC heads (training only, not used at inference), each RMSNorm + Linear upsample + GELU + Linear to blank + ACGT:
on the 12x stage output (x2: 12x has fewer frames than bases where the strand moves fast, as Dorado's CRF head) and
on the encoder output (48x, x8: ~4 bases per frame).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import apply_rope, rope_tables


@dataclass
class FastEncoderConfig:
    conv: tuple = ((16, 5, 1), (16, 5, 1), (32, 9, 3), (64, 9, 2), (128, 5, 2))  # (channels, kernel, stride)
    stages: tuple = ((128, 3, 2, 64), (256, 2, 4, 32), (512, 2, 8, 32))  # (width, layers, heads, window per side)
    ctc_hidden: int = 128
    head_dim: int = 64
    rope_theta: float = 10000.0
    max_len: int = 8192
    chunksize: int = 12288
    overlap: int = 600


def _ffn(d):  # SwiGLU hidden ~ 8/3 d, a multiple of 64
    return -(-8 * d // 3 // 64) * 64


class CTCHead(nn.Module):
    """[N, T, d] -> log-probabilities [N, up * T, 5] (blank, A, C, G, T)."""

    def __init__(self, d, up, hidden):
        super().__init__()
        self.up, self.hidden = up, hidden
        self.norm = nn.RMSNorm(d)
        self.proj = nn.Linear(d, up * hidden)
        self.out = nn.Linear(hidden, 5)

    def forward(self, h):
        N, T, _ = h.shape
        z = self.out(F.gelu(self.proj(self.norm(h)).view(N, self.up * T, self.hidden)))
        return z.float().log_softmax(-1)

    def loss(self, mid, targets, target_lens):
        """mid (frames [N, T, d], real frames per row); targets [N, L] base ids 1..4 (padded), target_lens [N].
        CTC loss summed over rows / all label bases of the batch (a window with few or no bases does not dominate)."""
        h, n = mid
        lp = self(h).transpose(0, 1)
        loss = F.ctc_loss(lp, targets, self.up * n, target_lens, blank=0, reduction="sum", zero_infinity=True)
        return loss / target_lens.sum().clamp(min=1)


class Layer(nn.Module):
    def __init__(self, d, heads, head_dim):
        super().__init__()
        self.h, self.hd = heads, head_dim
        self.attn_norm = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * heads * head_dim, bias=False)
        self.q_norm, self.k_norm = nn.RMSNorm(head_dim), nn.RMSNorm(head_dim)
        self.out = nn.Linear(heads * head_dim, d, bias=False)
        self.mlp_norm = nn.RMSNorm(d)
        self.gate_up = nn.Linear(d, 2 * _ffn(d), bias=False)
        self.down = nn.Linear(_ffn(d), d, bias=False)

    def forward(self, x, cos, sin, mask):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(B, T, 3, self.h, self.hd).unbind(2)
        q, k = apply_rope(self.q_norm(q), cos, sin), apply_rope(self.k_norm(k), cos, sin)
        q, k, v = (t.to(v.dtype).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, T, -1)
        x = x + self.out(o)
        gate, up = self.gate_up(self.mlp_norm(x)).chunk(2, -1)
        return x + self.down(F.silu(gate) * up)


class Stage(nn.Module):
    def __init__(self, d_in, d, n_layers, heads, window, head_dim):
        super().__init__()
        self.window = window
        self.down_norm = nn.RMSNorm(d_in) if d_in != d else None
        self.down = nn.Conv1d(d_in, d, 3, stride=2, padding=1) if d_in != d else None
        self.layers = nn.ModuleList([Layer(d, heads, head_dim) for _ in range(n_layers)])


class FastEncoder(nn.Module):
    """signal [N, 1, S] (standardised pA, as the Dorado encoder) -> frames [N, S / 48, 512]."""

    masked = True  # Basecaller.encode: forward takes per-row lengths

    def __init__(self, cfg: FastEncoderConfig = FastEncoderConfig(), mean=0.0, stdev=1.0):
        super().__init__()
        self.cfg = cfg
        convs, c, st = [], 1, 1
        for ch, k, s in cfg.conv:
            convs.append(nn.Conv1d(c, ch, k, stride=s, padding=k // 2))
            c, st = ch, st * s
        self.convs = nn.ModuleList(convs)
        self.conv_stride = st
        stages, d_in = [], c
        for i, (d, n, h, w) in enumerate(cfg.stages):
            assert d == h * cfg.head_dim and (i == 0) == (d_in == d)
            stages.append(Stage(d_in, d, n, h, w, cfg.head_dim))
            d_in = d
        self.stages = nn.ModuleList(stages)
        self.norm = nn.RMSNorm(d_in)
        self.d_model = d_in
        self.stride = st * 2 ** (len(stages) - 1)
        self.granularity = self.stride
        self.chunksize, self.overlap = cfg.chunksize, cfg.overlap
        self.mean, self.stdev = mean, stdev
        self.koi = None
        cos, sin = rope_tables(cfg.head_dim, cfg.max_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.ctc12 = CTCHead(cfg.stages[0][0], 2, cfg.ctc_hidden)
        self.ctc48 = CTCHead(d_in, 8, cfg.ctc_hidden)

    def forward(self, x, lens=None, return_mid=False):
        """x [N, 1, S]; lens [N] real samples per row (multiples of the stride; None: all S).
        Returns frames [N, S / stride, d_model] (zero past each row's frames); with return_mid also
        ((12x stage output [N, S / 12, 128], its frames per row), (frames, frames per row)) for the CTC heads."""
        N, _, S = x.shape
        lens = torch.full((N,), S, device=x.device) if lens is None else lens.to(x.device)

        def valid(T, st):  # [N, T] frames within the row
            return torch.arange(T, device=x.device)[None] < (lens[:, None] + st - 1) // st

        h, st = x * valid(S, 1)[:, None].to(x.dtype), 1
        for conv, (_, _, s) in zip(self.convs, self.cfg.conv):
            st *= s
            h = F.silu(conv(h))
            h = h * valid(h.shape[-1], st)[:, None].to(h.dtype)
        h = h.transpose(1, 2)
        mid = None
        for i, stage in enumerate(self.stages):
            if stage.down is not None:
                h = stage.down(stage.down_norm(h).transpose(1, 2)).transpose(1, 2)
                st *= 2
            T = h.shape[1]
            v = valid(T, st)
            t = torch.arange(T, device=x.device)
            band = (t[None, :] - t[:, None]).abs() <= stage.window
            # a padding query keeps its band (no all-masked rows); real queries see real keys only
            mask = (band[None] & (v[:, None, :] | ~v[:, :, None]))[:, None]
            for layer in stage.layers:
                h = layer(h, self.cos[:T], self.sin[:T], mask)
            h = h * v[..., None].to(h.dtype)
            if i == 0:
                mid = (h, v.sum(1))
        out = self.norm(h) * v[..., None].to(h.dtype)
        return (out, (mid, (out, v.sum(1)))) if return_mid else out


def ctc_greedy(logp, n):
    """[N, 2T, 5] log-probabilities, real upsampled frames per row -> base strings (best path, collapsed)."""
    best = logp.argmax(-1).cpu()
    out = []
    for b, m in zip(best, n.tolist()):
        p = b[:m]
        keep = torch.ones_like(p, dtype=torch.bool)
        keep[1:] = p[1:] != p[:-1]
        out.append("".join("NACGT"[i] for i in p[keep & (p > 0)].tolist()))
    return out


def ctc_targets(seqs, pad_to=1):
    """base strings -> (targets [N, L] long, 1..4 per base, 0 padding; L a multiple of pad_to; lengths [N])."""
    lut = torch.zeros(256, dtype=torch.long)
    for i, c in enumerate(b"ACGT"):
        lut[c] = i + 1
    L = max(-(-max(len(s) for s in seqs) // pad_to) * pad_to, 1)
    t = torch.zeros(len(seqs), L, dtype=torch.long)
    for b, s in enumerate(seqs):
        if s:  # an empty label (no bases in the window) stays all padding
            t[b, :len(s)] = lut[torch.frombuffer(bytearray(s.encode()), dtype=torch.uint8).long()]
    return t, torch.tensor([len(s) for s in seqs])

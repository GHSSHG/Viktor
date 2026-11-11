"""Mature Dorado encoder and optional 4x compression; all attention is sliding FlashAttention.

Adjacent pairs are ordered [even channels, odd channels]. Frame j at stride s
represents sample centre s*(j+.5), so a 12x centre c becomes (c+.5)/4-.5.
Padding is excluded before/after every local operation and from attention keys.
"""
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from .encoder import apply_rope, rope_tables

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


def mask_frames(x, lengths):
    return x.masked_fill(torch.arange(x.shape[1], device=x.device)[None, :, None] >= lengths[:, None, None], 0)


class LocalConv(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.depthwise = nn.Conv1d(width, width, 7, padding=3, groups=width)
        self.proj = nn.Linear(width, width)

    def forward(self, x, lengths):
        z = mask_frames(self.norm(x), lengths)
        z = F.silu(self.depthwise(z.transpose(1, 2)).transpose(1, 2))
        return mask_frames(x + self.proj(z), lengths)


class BottleneckTransformer(nn.Module):
    def __init__(self, width, window):
        super().__init__()
        self.window = window
        self.require_flash = True
        self.norm1, self.norm2 = nn.RMSNorm(width), nn.RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * 512, bias=False)
        self.out = nn.Linear(512, width, bias=False)
        self.qnorm, self.knorm = nn.RMSNorm(64), nn.RMSNorm(64)
        self.gate_up = nn.Linear(width, 2 * 1024, bias=False)
        self.down = nn.Linear(1024, width, bias=False)
        cos, sin = rope_tables(64, 4096, 10000.)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, x, lengths):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, 8, 64).unbind(2)
        q = apply_rope(self.qnorm(q).to(v.dtype), self.cos[:t], self.sin[:t])
        k = apply_rope(self.knorm(k).to(v.dtype), self.cos[:t], self.sin[:t])
        if flash_attn_varlen_func is not None and q.is_cuda and q.dtype in (torch.bfloat16, torch.float16):
            # Represent valid and padding regions as separate sequences: no masked
            # key can leak into the real prefix. Padding outputs are zeroed below.
            ends = (torch.arange(b, device=x.device, dtype=torch.int32) + 1) * t
            cu = torch.stack((ends - t + lengths.to(torch.int32), ends), -1).flatten()
            cu = F.pad(cu, (1, 0))
            y = flash_attn_varlen_func(q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1),
                                      cu, cu, t, t, window_size=(self.window, self.window)).view(b, t, 512)
        else:
            if self.require_flash:
                raise RuntimeError('Compression requires CUDA BF16/FP16 sliding FlashAttention')
            ix = torch.arange(t, device=x.device)
            allowed = ((ix[:, None] - ix[None, :]).abs() <= self.window)[None, None]
            allowed = allowed & (ix[None, None, None, :] < lengths[:, None, None, None])
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                               attn_mask=allowed).transpose(1, 2).reshape(b, t, 512)
        x = mask_frames(x + self.out(y), lengths)
        gate, up = self.gate_up(self.norm2(x)).chunk(2, -1)
        return mask_frames(x + self.down(F.silu(gate) * up), lengths)


class CompressionStage(nn.Module):
    def __init__(self, width, window):
        super().__init__()
        self.conv = LocalConv(width)
        self.transformer = BottleneckTransformer(width, window)

    def forward(self, x, lengths):
        x = mask_frames(x, lengths)
        if x.shape[1] % 2:
            x = F.pad(x, (0, 0, 0, 1))
        x = x.reshape(x.shape[0], x.shape[1] // 2, 2 * x.shape[2])
        lengths = (lengths + 1) // 2
        return self.transformer(self.conv(x, lengths), lengths), lengths


class MatureEncoder(nn.Module):
    def __init__(self, base, compression=4):
        super().__init__()
        assert compression in (1, 4)
        self.base, self.compression = base, compression
        self.gradient_checkpointing = True
        self.stages = nn.ModuleList([CompressionStage(1024, 64), CompressionStage(2048, 32)] if compression == 4 else [])
        self.d_model, self.stride = 512 * compression, base.stride * compression
        for name in ('mean', 'stdev', 'chunksize', 'granularity', 'overlap'):
            setattr(self, name, getattr(base, name))
        self.set_base_trainable(False)
        for layer in base.transformer_encoder:
            layer.self_attn.require_flash = True

    def set_base_trainable(self, value):
        self.base_trainable = value
        self.base.requires_grad_(value)
        self.base.grad_ckpt = value and self.gradient_checkpointing
        self.base.train(value and self.training)

    def train(self, mode=True):
        super().train(mode)
        if not self.base_trainable:
            self.base.eval()
        return self

    def compress(self, frames, lengths):
        for stage in self.stages:
            frames, lengths = stage(frames, lengths)
        return mask_frames(frames, lengths), lengths

    def base_forward(self, x):
        with nullcontext() if self.base_trainable else torch.no_grad():
            return self.base(x)

    def forward(self, x):
        frames = self.base_forward(x)
        lengths = torch.full((x.shape[0],), frames.shape[1], device=x.device, dtype=torch.long)
        return self.compress(frames, lengths)[0]

    def encode_batch(self, sig, valid):
        # Match original Dorado's repeat-padded short-chunk boundary, then remove
        # padding before the learned compression. Full windows keep a batched path.
        valid = valid.to(sig.device)
        full = valid == sig.shape[-1]
        if bool(full.all()):
            frames = self.base_forward(sig)
        else:
            buckets = {}
            for i, n in enumerate(valid.tolist()):
                padded = ((n + self.granularity - 1) // self.granularity) * self.granularity
                buckets.setdefault(padded, []).append(i)
            frames = None
            for padded, ids in buckets.items():
                idx = torch.tensor(ids, device=sig.device)
                rows = self.base_forward(sig.index_select(0, idx)[:, :, :padded])
                rows = F.pad(rows, (0, 0, 0, sig.shape[-1] // self.base.stride - rows.shape[1]))
                if frames is None:
                    frames = rows.new_zeros(sig.shape[0], rows.shape[1], rows.shape[2])
                frames = frames.index_copy(0, idx, rows)
        lengths = (valid + self.base.stride - 1) // self.base.stride
        return self.compress(frames, lengths)

    def encode_chunks(self, signal, lens, valid):
        out, lengths, off = [], [], 0
        for n, v in zip(lens, valid):
            frames, count = self.encode_batch(signal[off:off+n].view(1, 1, n), torch.tensor([v], device=signal.device))
            length = int(count[0])
            out.append(frames[0, :length])
            lengths.append(length)
            off += n
        return torch.cat(out), lengths

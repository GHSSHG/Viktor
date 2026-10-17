"""Dorado SUP transformer encoder (conv stack + windowed transformer) in PyTorch.

Mirrors dorado/nn/TxModules.cpp and ConvStack.cpp: conv + SiLU (batchnorm already
folded into the exported weights), NeoX-style RoPE (theta 10000), sliding-window
attention, SwiGLU = silu(second half) * first half, DeepNorm residual + RMSNorm (eps 1e-5).
The upsample and CRF head are not loaded: the decoder consumes the transformer output.
"""

import glob
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


def rope_tables(dim: int, max_len: int, theta: float):
    inv = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    freqs = torch.arange(max_len, dtype=torch.float64)[:, None] * inv
    return freqs.cos().float(), freqs.sin().float()


def apply_rope(x, cos, sin):
    """x: [B, T, H, D]; cos/sin: [T, D/2]. Rotates first half against second half."""
    x1, x2 = x.float().chunk(2, -1)
    cos, sin = cos[:, None], sin[:, None]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).to(x.dtype)


class Conv(nn.Module):
    def __init__(self, insize, size, winlen, stride, padding, bias):
        super().__init__()
        self.conv = nn.Conv1d(insize, size, winlen, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return F.silu(self.conv(x))


class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead, window, theta, max_len):
        super().__init__()
        self.nhead, self.head_dim = nhead, d_model // nhead
        self.window = tuple(window)  # (left, right)
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        cos, sin = rope_tables(self.head_dim, max_len, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self._mask = None

    def band_mask(self, T, device):
        if self._mask is None or self._mask.shape[0] != T or self._mask.device != device:
            i = torch.arange(T, device=device)
            rel = i[None, :] - i[:, None]
            self._mask = (rel >= -self.window[0]) & (rel <= self.window[1])
        return self._mask

    def forward(self, x):
        N, T, C = x.shape
        q, k, v = self.Wqkv(x).view(N, T, 3, self.nhead, self.head_dim).unbind(2)
        cos, sin = self.cos[:T], self.sin[:T]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if flash_attn_func is not None and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            o = flash_attn_func(q, k, v, window_size=self.window)
        else:
            o = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                attn_mask=self.band_mask(T, x.device)).transpose(1, 2)
        return self.out_proj(o.reshape(N, T, C))


class GatedMlp(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 2 * hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        y, gate = self.fc1(x).chunk(2, -1)
        return self.fc2(F.silu(gate) * y)


class TxLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, deepnorm_alpha, attn_window, theta, max_len):
        super().__init__()
        self.self_attn = SelfAttention(d_model, nhead, attn_window, theta, max_len)
        self.ff = GatedMlp(d_model, dim_feedforward)
        self.norm1 = nn.RMSNorm(d_model, eps=1e-5)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-5)
        self.register_buffer("deepnorm_alpha", torch.tensor(float(deepnorm_alpha)))

    def forward(self, x):
        x = self.norm1(self.self_attn(x) + self.deepnorm_alpha * x)
        return self.norm2(self.ff(x) + self.deepnorm_alpha * x)


class DoradoEncoder(nn.Module):
    """signal [N, 1, T] (standardised pA) -> frames [N, T / stride, d_model]."""

    def __init__(self, cfg: dict):
        super().__init__()
        enc = cfg["model"]["encoder"]
        convs = [l for l in enc["conv"]["sublayers"] if l["type"] == "convolution"]
        self.conv = nn.Sequential(*[
            Conv(l["insize"], l["size"], l["winlen"], l["stride"], l["padding"], l["bias"]) for l in convs])
        self.stride = 1
        for l in convs:
            self.stride *= l["stride"]
        tx = enc["transformer_encoder"]
        lay = tx["layer"]
        self.d_model = lay["d_model"]
        self.transformer_encoder = nn.Sequential(*[
            TxLayer(lay["d_model"], lay["nhead"], lay["dim_feedforward"], lay["deepnorm_alpha"],
                    lay["attn_window"], lay.get("theta", 10000.0), lay.get("max_seq_len", 4096))
            for _ in range(tx["depth"])])
        std = cfg.get("standardisation", {})
        self.mean, self.stdev = std.get("mean", 0.0), std.get("stdev", 1.0)
        self.chunksize = cfg["basecaller"]["chunksize"]
        self.overlap = cfg["basecaller"]["overlap"]

    def standardise(self, pa):
        return (pa - self.mean) / self.stdev

    def forward(self, x):
        return self.transformer_encoder(self.conv(x).transpose(1, 2))


def _load_tensor(path):
    return next(torch.jit.load(path, map_location="cpu").parameters()).detach()


def load_dorado_encoder(model_dir: str, load_weights: bool = True) -> DoradoEncoder:
    """Build the encoder from a Dorado model directory and load its released weights."""
    with open(os.path.join(model_dir, "config.toml"), "rb") as f:
        cfg = tomllib.load(f)
    enc = DoradoEncoder(cfg)
    if load_weights:
        sd = {}
        for path in glob.glob(os.path.join(model_dir, "*.tensor")):
            name = os.path.basename(path)[:-len(".tensor")]
            if name.startswith(("conv.", "transformer_encoder.")):
                sd[name] = _load_tensor(path)
        enc.load_state_dict(sd, strict=True)
    return enc

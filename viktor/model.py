"""Basecaller = Dorado encoder (released weights) + per-frame projector + decoder-only LM."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import Decoder, DecoderConfig
from .encoder import DoradoEncoder
from .tokenizer import PAD, KmerTokenizer


@dataclass
class AlignConfig:
    """Attention alignment supervision (training only): in each listed layer, the listed query heads' attention
    over the signal prefix, at the position that predicts base i, is pulled (KL) towards a Gaussian of width sigma
    frames around base i's frame from the forced alignment. Only the attention restricted to the prefix is supervised."""
    layers: tuple = (2, 4)
    heads: tuple = (0, 4)
    sigma: float = 1.0
    weight: float = 0.1


class Basecaller(nn.Module):
    def __init__(self, encoder: DoradoEncoder, tokenizer: KmerTokenizer, cfg: Optional[DecoderConfig] = None):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.cfg = cfg or DecoderConfig(vocab_size=tokenizer.vocab_size)
        d = self.cfg.d_model
        # No frame compression: one prefix token per encoder frame (1024 per 12288-sample chunk).
        self.projector = nn.Sequential(nn.Linear(encoder.d_model, d), nn.GELU(), nn.Linear(d, d))
        self.decoder = Decoder(self.cfg)
        self.align: Optional[AlignConfig] = None

    def set_encoder_trainable(self, trainable: bool):
        if trainable and getattr(self.encoder, "koi", None) is not None:
            raise RuntimeError("the koi encoder is inference-only (no gradients); train the PyTorch encoder instead")
        self.encoder.requires_grad_(trainable)

    def logits(self, frames, tokens_in):
        """Teacher forcing from encoder frames [B, T, d_enc]: logits for every token position, [B, T_tok, vocab]."""
        prefix = self.projector(frames.to(self.projector[0].weight.dtype))
        x = torch.cat([prefix, self.decoder.embed(tokens_in)], 1)
        return self.decoder.head(self.decoder(x)[:, prefix.shape[1]:])

    def forward(self, signal, tokens_in):
        return self.logits(self.encoder(signal), tokens_in)

    def batch(self, seqs, weights=None, pad_to=1, centers=None):
        """CPU tensors for teacher forcing: tokens_in, targets [B, T] and target weights [B, T], T padded up to a
        multiple of pad_to (fewer distinct shapes for torch.compile). weights (single-base tokenizer): one array
        per sequence, a weight per base; EOS has weight 1. Without weights every target (incl. EOS) weighs 1.
        centers (with weights): one array per sequence, the prefix frame of each base; adds a fourth tensor [B, T],
        -1 where there is no alignment target (EOS, padding, zero-weight bases)."""
        tokens_in, targets, _ = self.tokenizer.batch(seqs)
        T = -(-targets.shape[1] // pad_to) * pad_to
        tokens_in = F.pad(tokens_in, (0, T - tokens_in.shape[1]), value=PAD)
        targets = F.pad(targets, (0, T - targets.shape[1]), value=PAD)
        if weights is None:
            return tokens_in, targets, (targets != PAD).float()
        assert self.tokenizer.k == 1, "per-base weights need the single-base tokenizer"
        w = torch.zeros(targets.shape)
        for b, wb in enumerate(weights):
            w[b, :len(wb)] = torch.as_tensor(wb)
            w[b, len(wb)] = 1.0  # EOS
        if centers is None:
            return tokens_in, targets, w
        c = torch.full(targets.shape, -1.0)
        for b, (cb, wb) in enumerate(zip(centers, weights)):
            c[b, :len(cb)] = torch.where(torch.as_tensor(wb) > 0, torch.as_tensor(cb), -1.0)
        return tokens_in, targets, w, c

    def weighted_loss(self, frames, tokens_in, targets, w, centers=None):
        """Weighted-mean token cross-entropy (tensors only, compilable), plus self.align's attention alignment loss
        when centers are given. Returns (loss, ce, align, correct, scored): loss = ce + weight * align; greedy-correct
        and scored targets counted where the weight is positive."""
        prefix = self.projector(frames.to(self.projector[0].weight.dtype))
        P = prefix.shape[1]
        x = torch.cat([prefix, self.decoder.embed(tokens_in)], 1)
        if centers is None or self.align is None:
            h, qk = self.decoder(x), []
        else:
            h, qk = self.decoder.forward_probe(x, self.align.layers)
        logits = self.decoder.head(h[:, P:])
        ce = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=PAD,
                             reduction="none").view_as(targets)
        scored = w > 0
        ce = (ce * w).sum() / w.sum()
        correct = ((logits.argmax(-1) == targets) & scored).sum()
        if not qk:
            return ce, ce, torch.zeros_like(ce), correct, scored.sum()
        align = self.align_loss(qk, centers, P)
        return ce + self.align.weight * align, ce, align, correct, scored.sum()

    def align_loss(self, qk, centers, P):
        """Mean KL(target || attention over the P prefix frames) over aligned positions, heads and layers."""
        a = self.align
        mask = centers >= 0
        frame = torch.arange(P, device=centers.device, dtype=torch.float32)
        tgt = torch.log_softmax(-(frame - centers[..., None]).square() / (2 * a.sigma ** 2), -1)[:, None]  # [B,1,T,P]
        heads = list(a.heads)
        kv = [h * self.cfg.n_kv_heads // self.cfg.n_heads for h in heads]
        total = 0.0
        for q, k in qk:
            s = torch.einsum("bthd,bphd->bhtp", q[:, P:, heads], k[:, :P, kv]) * q.shape[-1] ** -0.5
            kl = (tgt.exp() * (tgt - torch.log_softmax(s.float(), -1))).sum(-1)  # [B, heads, T]
            total = total + (kl * mask[:, None]).sum() / (mask.sum() * len(heads))
        return total / len(qk)

    def frame_loss(self, frames, seqs, weights=None):
        dev = frames.device
        return self.weighted_loss(frames, *(t.to(dev) for t in self.batch(seqs, weights)))

    def loss(self, signal, seqs, weights=None):
        return self.frame_loss(self.encoder(signal), seqs, weights)[0]

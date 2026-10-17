"""Basecaller = Dorado encoder (released weights) + per-frame projector + decoder-only LM."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import Decoder, DecoderConfig
from .encoder import DoradoEncoder
from .tokenizer import PAD, KmerTokenizer


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

    def batch(self, seqs, weights=None, pad_to=1):
        """CPU tensors for teacher forcing: tokens_in, targets [B, T] and target weights [B, T], T padded up to a
        multiple of pad_to (fewer distinct shapes for torch.compile). weights (single-base tokenizer): one array
        per sequence, a weight per base; EOS has weight 1. Without weights every target (incl. EOS) weighs 1."""
        tokens_in, targets, _ = self.tokenizer.batch(seqs)
        T = -(-targets.shape[1] // pad_to) * pad_to
        tokens_in = F.pad(tokens_in, (0, T - tokens_in.shape[1]), value=PAD)
        targets = F.pad(targets, (0, T - targets.shape[1]), value=PAD)
        if weights is None:
            w = (targets != PAD).float()
        else:
            assert self.tokenizer.k == 1, "per-base weights need the single-base tokenizer"
            w = torch.zeros(targets.shape)
            for b, wb in enumerate(weights):
                w[b, :len(wb)] = torch.as_tensor(wb)
                w[b, len(wb)] = 1.0  # EOS
        return tokens_in, targets, w

    def weighted_loss(self, frames, tokens_in, targets, w):
        """Weighted-mean token cross-entropy (tensors only, compilable). Returns (loss, correct, scored):
        greedy-correct and scored targets counted where the weight is positive."""
        logits = self.logits(frames, tokens_in)
        ce = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=PAD,
                             reduction="none").view_as(targets)
        scored = w > 0
        return (ce * w).sum() / w.sum(), ((logits.argmax(-1) == targets) & scored).sum(), scored.sum()

    def frame_loss(self, frames, seqs, weights=None):
        dev = frames.device
        return self.weighted_loss(frames, *(t.to(dev) for t in self.batch(seqs, weights)))

    def loss(self, signal, seqs, weights=None):
        return self.frame_loss(self.encoder(signal), seqs, weights)[0]

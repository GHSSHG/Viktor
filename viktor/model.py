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
        self.encoder.requires_grad_(trainable)

    def forward(self, signal, tokens_in):
        """Teacher forcing: logits for every token position, [B, T_tok, vocab]."""
        frames = self.encoder(signal)
        prefix = self.projector(frames.to(self.projector[0].weight.dtype))
        x = torch.cat([prefix, self.decoder.embed(tokens_in)], 1)
        return self.decoder.head(self.decoder(x)[:, prefix.shape[1]:])

    def loss(self, signal, seqs):
        """Token cross-entropy summed and normalised by the number of bases, so k=1 and k=4 are comparable."""
        tokens_in, targets, n_bases = self.tokenizer.batch(seqs)
        tokens_in, targets = tokens_in.to(signal.device), targets.to(signal.device)
        logits = self(signal, tokens_in)
        ce = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=PAD, reduction="sum")
        return ce / n_bases

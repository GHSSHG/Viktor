"""Basecaller = Dorado encoder (released weights) + per-frame projector + decoder-only LM."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
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

    @torch.no_grad()
    def encode(self, sig, valid):
        """Decoder input as at inference: encoder frames [B, P, d_enc] (P = frames of a full window) and the real
        frames per window [B]. sig [B, 1, S] from data.collate, valid: real samples per window; a short window is
        encoded at its repeat-padded length (encoder.encode_chunks) and its frames past the real signal are zero."""
        S, st, g = sig.shape[-1], self.encoder.stride, self.encoder.granularity
        if bool((valid == S).all()):
            frames = self.encoder(sig)
            return frames, torch.full((len(valid),), frames.shape[1], device=frames.device)
        lens = [-(-int(v) // g) * g for v in valid]
        flat, flens = self.encoder.encode_chunks(torch.cat([sig[b, 0, :n] for b, n in enumerate(lens)]), lens,
                                                 [int(v) for v in valid])
        frames = flat.new_zeros(len(lens), S // st, flat.shape[-1])
        for b, (o, n) in enumerate(zip(np.cumsum([0] + flens[:-1]), flens)):
            frames[b, :n] = flat[o:o + n]
        return frames, torch.tensor(flens, device=frames.device)

    def batch(self, seqs, weights=None, pad_to=1, centers=None):
        """CPU tensors for teacher forcing: tokens_in, targets [B, T] and target weights [B, T], T padded up to a
        multiple of pad_to (fewer distinct shapes for torch.compile). weights: one array per sequence, a weight per
        base; a k-mer token weighs its lowest base weight (a masked base masks its token), EOS 1. Without weights
        every target (incl. EOS) weighs 1. centers (with weights): one array per sequence, the prefix frame of each
        base; adds a fourth tensor [B, T], a token's mean base frame, -1 where there is no alignment target (EOS,
        padding, tokens with a zero-weight or unaligned base)."""
        tokens_in, targets, _ = self.tokenizer.batch(seqs)
        T = -(-targets.shape[1] // pad_to) * pad_to
        tokens_in = F.pad(tokens_in, (0, T - tokens_in.shape[1]), value=PAD)
        targets = F.pad(targets, (0, T - targets.shape[1]), value=PAD)
        if weights is None:
            return tokens_in, targets, (targets != PAD).float()
        k = self.tokenizer.k

        def per_token(a, fill):  # [n bases] -> [n tokens, k]
            out = np.full(-(-len(a) // k) * k, fill, np.float32)
            out[:len(a)] = a
            return out.reshape(-1, k)

        w = torch.zeros(targets.shape)
        for b, wb in enumerate(weights):
            wt = per_token(wb, np.inf).min(1)
            w[b, :len(wt)] = torch.from_numpy(wt)
            w[b, len(wt)] = 1.0  # EOS
        if centers is None:
            return tokens_in, targets, w
        c = torch.full(targets.shape, -1.0)
        for b, (cb, wb) in enumerate(zip(centers, weights)):
            ok = (per_token(wb, 1.0) > 0).all(1) & (per_token(cb, 0.0) >= 0).all(1)
            ct = np.nanmean(per_token(cb, np.nan), 1) if len(cb) else np.zeros(0, np.float32)
            c[b, :len(ct)] = torch.from_numpy(np.where(ok, ct, -1.0).astype(np.float32))
        return tokens_in, targets, w, c

    def weighted_loss(self, frames, tokens_in, targets, w, centers=None, plens=None):
        """Weighted-mean token cross-entropy (tensors only, compilable), plus self.align's attention alignment loss
        when centers are given. plens [B]: real prefix frames per window (Basecaller.encode); window b's sequence is
        [its plens[b] frames | tokens | padding], as decoded at inference (causal: the padding is never attended).
        Returns (loss, ce, align, correct, scored): loss = ce + weight * align; greedy-correct and scored targets
        counted where the weight is positive."""
        prefix = self.projector(frames.to(self.projector[0].weight.dtype))
        B, P, d = prefix.shape
        T = tokens_in.shape[1]
        emb = self.decoder.embed(tokens_in)
        if plens is None:
            x, at = torch.cat([prefix, emb], 1), torch.arange(P, P + T, device=prefix.device).expand(B, T)
        else:
            pos = torch.arange(P + T, device=prefix.device).expand(B, -1)
            tok = pos >= plens[:, None]
            x = torch.where(tok[..., None], emb.gather(1, (pos - plens[:, None]).clamp(0, T - 1)[..., None].expand(-1, -1, d)),
                            prefix.gather(1, pos.clamp(max=P - 1)[..., None].expand(-1, -1, d)))
            at = plens[:, None] + torch.arange(T, device=prefix.device)  # sequence position of each token
        if centers is None or self.align is None:
            h, qk = self.decoder(x), []
        else:
            h, qk = self.decoder.forward_probe(x, self.align.layers)
        logits = self.decoder.head(h.gather(1, at[..., None].expand(-1, -1, h.shape[-1])))
        ce = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=PAD,
                             reduction="none").view_as(targets)
        scored = w > 0
        ce = (ce * w).sum() / w.sum()
        correct = ((logits.argmax(-1) == targets) & scored).sum()
        if not qk:
            return ce, ce, torch.zeros_like(ce), correct, scored.sum()
        align = self.align_loss(qk, centers, P, at, plens)
        return ce + self.align.weight * align, ce, align, correct, scored.sum()

    def align_loss(self, qk, centers, P, at, plens=None):
        """Mean KL(target || attention over the real prefix frames) over aligned positions, heads and layers.
        at [B, T]: sequence position of each token."""
        a = self.align
        mask = centers >= 0
        frame = torch.arange(P, device=centers.device, dtype=torch.float32)
        # a short window's prefix ends before P (later positions are tokens): neither target nor attention there
        out = torch.zeros(1, 1, 1, P, dtype=torch.bool, device=centers.device) if plens is None else \
            frame[None, None, None] >= plens[:, None, None, None]  # [B|1, 1, 1, P]
        tgt = -(frame - centers[..., None]).square() / (2 * a.sigma ** 2)
        tgt = torch.log_softmax(tgt[:, None].masked_fill(out, float("-inf")), -1)  # [B, 1, T, P]
        heads = list(a.heads)
        kv = [h * self.cfg.n_kv_heads // self.cfg.n_heads for h in heads]
        total = 0.0
        for q, k in qk:
            qt = q[:, :, heads].gather(1, at[..., None, None].expand(-1, -1, len(heads), q.shape[-1]))
            s = torch.einsum("bthd,bphd->bhtp", qt, k[:, :P, kv]) * q.shape[-1] ** -0.5
            logq = torch.log_softmax(s.float().masked_fill(out, float("-inf")), -1)
            kl = (tgt.exp() * (tgt - logq)).masked_fill(out, 0.0).sum(-1)  # [B, heads, T]
            total = total + (kl * mask[:, None]).sum() / (mask.sum() * len(heads))
        return total / len(qk)

    def frame_loss(self, frames, seqs, weights=None, plens=None):
        dev = frames.device
        return self.weighted_loss(frames, *(t.to(dev) for t in self.batch(seqs, weights)), plens=plens)

    def loss(self, signal, seqs, weights=None):
        return self.frame_loss(self.encoder(signal), seqs, weights)[0]

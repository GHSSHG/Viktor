"""Optimised greedy decoding: static KV cache, batched prefill, one CUDA graph per decode step.

Every chunk has the same prefix length (n_frames + BOS), so all sequences in a batch sit at the
same position each step; only their stop times differ. Finished sequences keep running and emit
PAD, which keeps every step shape-static and graph-capturable.
"""

import torch

from .decoder import KVCache
from .model import Basecaller
from .tokenizer import BOS, EOS, PAD


class DecodeEngine:
    def __init__(self, model: Basecaller, batch_size: int, max_new_tokens: int, n_frames: int,
                 cuda_graph: bool = True, compile: bool = False, prefill_batch: int = 128,
                 check_every: int = 32):
        self.model, self.dec = model, model.decoder
        dev = self.dec.lm_head.weight.device
        self.dtype = self.dec.lm_head.weight.dtype
        self.B, self.max_new = batch_size, max_new_tokens
        self.prefix_len = n_frames + 1
        self.prefill_batch, self.check_every = prefill_batch, check_every
        assert self.prefix_len + max_new_tokens <= self.dec.cfg.max_seq_len
        self.cache = KVCache(self.dec.cfg, batch_size, self.prefix_len + max_new_tokens, self.dtype, dev)

        self.tok = torch.full((batch_size,), BOS, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.seqlens = torch.zeros(batch_size, dtype=torch.int32, device=dev)
        self.out = torch.full((batch_size, max_new_tokens), PAD, dtype=torch.long, device=dev)
        self.bos = torch.tensor([BOS], device=dev)
        V = model.tokenizer.vocab_size
        self.bias = torch.zeros(V, device=dev)
        self.bias[[PAD, BOS]] = float("-inf")
        self.is_tail = model.tokenizer.tail_mask.to(dev)

        self.step_fn = torch.compile(self._step) if compile else self._step
        self.graph = None
        if cuda_graph and dev.type == "cuda":
            self._capture()

    def _select(self, logits, prev):
        nxt = (logits.float() + self.bias).argmax(-1)
        nxt = torch.where(self.is_tail[prev], torch.full_like(nxt, EOS), nxt)  # a tail must be last
        stop = (prev == EOS) | (prev == PAD)
        return torch.where(stop, torch.full_like(nxt, PAD), nxt)

    def _step(self):
        self.seqlens.copy_(self.pos.to(torch.int32).expand(self.B))
        h = self.dec(self.dec.embed(self.tok)[:, None], cache=self.cache, pos=self.pos, seqlens=self.seqlens)
        nxt = self._select(self.dec.head(h[:, 0]), self.tok)
        self.out.index_copy_(1, self.pos - (self.prefix_len - 1), nxt[:, None])
        self.tok.copy_(nxt)
        self.pos.add_(1)

    @torch.no_grad()
    def _capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.pos.fill_(self.prefix_len)
                self.step_fn()
        torch.cuda.current_stream().wait_stream(s)
        self.pos.fill_(self.prefix_len)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.step_fn()

    @torch.no_grad()
    def prefill(self, frames):
        """frames: [B, n_frames, d_enc] encoder output. Fills the cache and emits the first token."""
        assert frames.shape[0] == self.B and frames.shape[1] + 1 == self.prefix_len
        bos = self.dec.embed(self.bos)[None]
        last = []
        for lo in range(0, self.B, self.prefill_batch):
            hi = min(self.B, lo + self.prefill_batch)
            x = torch.cat([self.model.projector(frames[lo:hi].to(self.dtype)), bos.expand(hi - lo, 1, -1)], 1)
            last.append(self.dec(x, cache=self.cache, batch_slice=slice(lo, hi))[:, -1])
        self.tok.fill_(BOS)
        first = self._select(self.dec.head(torch.cat(last)), self.tok)
        self.out.fill_(PAD)
        self.out[:, 0] = first
        self.tok.copy_(first)
        self.pos.fill_(self.prefix_len)

    @torch.no_grad()
    def decode(self, steps=None):
        """Run decode steps after prefill. steps=None: stop when all sequences ended (checked every
        check_every steps) or at max_new_tokens. steps=n: run exactly n-1 steps (n output tokens)."""
        n = self.max_new - 1 if steps is None else steps - 1
        for i in range(n):
            if self.graph is not None:
                self.graph.replay()
            else:
                self.step_fn()
            if steps is None and (i + 1) % self.check_every == 0 and bool(((self.tok == EOS) | (self.tok == PAD)).all()):
                break
        return self.out

    def generate(self, frames, steps=None):
        """Returns token ids [B, max_new_tokens] (PAD after EOS). The buffer is reused by the next call."""
        self.prefill(frames)
        return self.decode(steps)

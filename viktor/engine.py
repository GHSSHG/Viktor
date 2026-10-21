"""Greedy (or sampled, temperature > 0) decoding for variable-length chunks: static KV cache, packed prefill, one CUDA
graph per step. Sampling uses the Gumbel-max trick (argmax of logits / temperature + Gumbel noise), graph-safe.

Each sequence's prefix is the encoder frames of its chunk's real signal + BOS, so prefix lengths differ
and every sequence decodes at its own position. Finished sequences keep running and emit PAD, which
keeps every step shape-static and graph-capturable. Prefill keeps all K/V but runs the last layer for
BOS only (the signal positions' final outputs have no consumer).

backend "triton": fused kernels (fast_decoder.py), CUDA only; attention_backend picks the decode
attention ("auto" = decode_attention.py on sm80, else FlashAttention). backend "torch": reference decoder.
"""

import torch

from .decoder import KVCache
from .model import Basecaller
from .tokenizer import BOS, EOS, PAD


def _triton_available():
    try:
        import flash_attn  # noqa: F401
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


class DecodeEngine:
    def __init__(self, model: Basecaller, batch_size: int, max_new_tokens: int, max_frames: int,
                 backend: str = "auto", cuda_graph: bool = True, prefill_batch: int = 128, check_every: int = 32,
                 attention_backend: str = "auto", temperature: float = 0.0):
        self.model, self.dec = model, model.decoder
        self.temperature = temperature
        dev = self.dec.lm_head.weight.device
        self.dtype = self.dec.lm_head.weight.dtype
        if backend == "auto":
            backend = "triton" if dev.type == "cuda" and _triton_available() else "torch"
        self.backend = backend
        if backend == "triton":
            from .fast_decoder import FastDecoder
            self.fast = FastDecoder(self.dec, attention_backend=attention_backend)
        self.B, self.max_new = batch_size, max_new_tokens
        self.prefill_batch, self.check_every = prefill_batch, check_every
        self._prefill_plans = {}
        max_len = max_frames + 1 + max_new_tokens
        assert max_len <= self.dec.cfg.max_seq_len
        self.cache = KVCache(self.dec.cfg, batch_size, max_len, self.dtype, dev)

        self.tok = torch.full((batch_size,), BOS, dtype=torch.long, device=dev)
        self.pos = torch.zeros(batch_size, dtype=torch.long, device=dev)
        self.pos32 = torch.zeros(batch_size, dtype=torch.int32, device=dev)
        self.seqlens = torch.zeros(batch_size, dtype=torch.int32, device=dev)
        self.rows = torch.arange(batch_size, dtype=torch.int32, device=dev)
        self.step_idx = torch.zeros(1, dtype=torch.long, device=dev)
        self.out = torch.full((batch_size, max_new_tokens), PAD, dtype=torch.long, device=dev)
        self.bos = torch.tensor([BOS], device=dev)
        self.bias = torch.zeros(model.tokenizer.vocab_size, device=dev)
        self.bias[[PAD, BOS]] = float("-inf")
        self.is_tail = model.tokenizer.tail_mask.to(dev)

        self.graph = None
        if cuda_graph and dev.type == "cuda":
            self._capture()

    def _select(self, logits, prev):
        z = logits.float() + self.bias
        if self.temperature > 0:
            z = z / self.temperature - torch.log(-torch.log(torch.rand_like(z).clamp_(min=1e-20)))
        nxt = z.argmax(-1)
        nxt = torch.where(self.is_tail[prev], torch.full_like(nxt, EOS), nxt)  # a tail must be last
        stop = (prev == EOS) | (prev == PAD)
        return torch.where(stop, torch.full_like(nxt, PAD), nxt)

    def _step(self):
        x = self.dec.embed(self.tok)
        if self.backend == "triton":
            self.pos32.copy_(self.pos)
            self.seqlens.copy_(self.pos + 1)
            h = self.fast.decode(x, self.pos32, self.rows, self.seqlens, self.cache)
        else:
            self.seqlens.copy_(self.pos)
            h = self.dec(x[:, None], cache=self.cache, pos=self.pos, seqlens=self.seqlens)[:, 0]
        nxt = self._select(self.dec.head(h), self.tok)
        self.out.index_copy_(1, self.step_idx, nxt[:, None])
        self.tok.copy_(nxt)
        self.pos.add_(1)
        self.step_idx.add_(1)

    def _reset_state(self):
        self.pos.fill_(1)
        self.step_idx.zero_()

    @torch.no_grad()
    def _capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._reset_state()
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self._reset_state()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()

    def _prefill_triton(self, prefix, lens, row0):
        """prefix [sum(lens), d] projected frames of sequences row0.. ; returns hidden at each BOS."""
        dev = prefix.device
        n = len(lens)
        # One layout per subbatch, replaced when its lengths change. Sizes
        # come from CPU metadata, so construction never reads GPU scalars.
        key = tuple(lens)
        entry = self._prefill_plans.get(row0)
        if entry is None or entry[0] != key:
            offsets = [0]
            for length in lens:
                offsets.append(offsets[-1] + length + 1)
            total, frames = offsets[-1], sum(lens)
            cu = torch.tensor(offsets, dtype=torch.int32, device=dev)
            L = torch.tensor(lens, device=dev)
            frame_seg = torch.repeat_interleave(torch.arange(n, device=dev), L, output_size=frames)
            frame_idx = torch.arange(frames, device=dev) + frame_seg
            last = cu[1:].long() - 1
            pos = (torch.arange(total, device=dev) - torch.repeat_interleave(cu[:-1].long(), L + 1,
                                                                           output_size=total)).int()
            seq = torch.repeat_interleave(torch.arange(row0, row0 + n, device=dev, dtype=torch.int32),
                                          L + 1, output_size=total)
            cuq = torch.arange(n + 1, device=dev, dtype=torch.int32)
            entry = key, (frame_idx, last, pos, seq, cu, cuq, total, max(lens) + 1)
            self._prefill_plans[row0] = entry
        frame_idx, last, pos, seq, cu, cuq, total, max_len = entry[1]
        x = torch.empty(total, prefix.shape[1], dtype=self.dtype, device=dev)
        x[frame_idx] = prefix
        x[last] = self.dec.embed(self.bos).to(self.dtype)
        return self.fast.prefill(x, pos, seq, cu, max_len, self.cache, last_indices=last, cu_queries=cuq)

    @torch.no_grad()
    def prefill(self, frames, frame_lens):
        """frames [sum(frame_lens), d_enc]: encoder frames of B chunks, concatenated; frame_lens [B].
        Fills the cache and emits the first token of every sequence."""
        frame_lens = [int(n) for n in frame_lens]
        assert len(frame_lens) == self.B and frames.shape[0] == sum(frame_lens)
        offs = [0]
        for n in frame_lens:
            offs.append(offs[-1] + n)
        last = []
        for lo in range(0, self.B, self.prefill_batch):
            hi = min(self.B, lo + self.prefill_batch)
            prefix = self.model.projector(frames[offs[lo]:offs[hi]].to(self.dtype))
            if self.backend == "triton":
                last.append(self._prefill_triton(prefix, frame_lens[lo:hi], lo))
            else:
                bos = self.dec.embed(self.bos)[None]
                for b in range(lo, hi):
                    x = torch.cat([prefix[offs[b] - offs[lo]:offs[b + 1] - offs[lo]][None], bos], 1)
                    last.append(self.dec(x, cache=self.cache, batch_slice=slice(b, b + 1))[:, -1])
        self.tok.fill_(BOS)
        first = self._select(self.dec.head(torch.cat(last)), self.tok)
        self.out.fill_(PAD)
        self.out[:, 0] = first
        self.tok.copy_(first)
        self.pos.copy_(torch.tensor(frame_lens, device=self.pos.device) + 1)
        self.step_idx.fill_(1)

    @torch.no_grad()
    def decode(self, steps=None):
        """Run decode steps after prefill. steps=None: stop when all sequences ended (checked every
        check_every steps) or at max_new_tokens. steps=n: run exactly n-1 steps (n output tokens)."""
        n = self.max_new - 1 if steps is None else steps - 1
        for i in range(n):
            if self.graph is not None:
                self.graph.replay()
            else:
                self._step()
            if steps is None and (i + 1) % self.check_every == 0 and bool(((self.tok == EOS) | (self.tok == PAD)).all()):
                break
        return self.out

    def generate(self, frames, frame_lens, steps=None):
        """Returns token ids [B, max_new_tokens] (PAD after EOS). The buffer is reused by the next call."""
        self.prefill(frames, frame_lens)
        return self.decode(steps)

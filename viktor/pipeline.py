"""Encoder/decoder pipelining across batches on one GPU.

While batch i is decoded on the (high-priority) decoder stream, batch i+1 is encoded on the
encoder stream. overlap=False runs the same work strictly in sequence for comparison.
A batch is (signal, lens, valid): Dorado-style chunks concatenated, lens = samples per padded
chunk, valid = real samples per chunk (the decoder sees only the frames of real signal).
"""

import contextlib

import torch

from .engine import DecodeEngine
from .model import Basecaller


class Pipeline:
    def __init__(self, model: Basecaller, engine: DecodeEngine, high_priority_decoder: bool = True):
        self.model, self.engine = model, engine
        self.cuda = engine.out.is_cuda
        if self.cuda:
            self.enc_stream = torch.cuda.Stream()
            self.dec_stream = torch.cuda.Stream(priority=-1 if high_priority_decoder else 0)

    @torch.no_grad()
    def _encode(self, batch):
        if not self.cuda:
            return self.model.encoder.encode_chunks(*batch), None
        self.enc_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.enc_stream):
            enc = self.model.encoder.encode_chunks(*batch)
            ev = torch.cuda.Event()
            ev.record()
        return enc, ev

    @torch.no_grad()
    def _decode(self, item, steps):
        (frames, frame_lens), ev = item
        ctx = contextlib.nullcontext()
        if self.cuda:
            ctx = torch.cuda.stream(self.dec_stream)
        with ctx:
            if self.cuda:
                self.dec_stream.wait_event(ev)
                frames.record_stream(self.dec_stream)
            # .to(cpu) synchronises the decoder stream only
            return self.engine.generate(frames, frame_lens, steps).to("cpu", copy=True)

    @torch.no_grad()
    def run(self, batches, steps=None, overlap=True):
        """batches: iterable of (signal, lens, valid). steps: int or one int per batch. Returns token tensors."""
        batches = list(batches)
        steps = steps if isinstance(steps, (list, tuple)) else [steps] * len(batches)
        out = []
        if not overlap:
            for batch, st in zip(batches, steps):
                frames, frame_lens = self.model.encoder.encode_chunks(*batch)
                out.append(self.engine.generate(frames, frame_lens, st).to("cpu", copy=True))
            return out
        cur = self._encode(batches[0])
        for i in range(1, len(batches)):
            nxt = self._encode(batches[i])  # queued before decode(cur) blocks, so it runs alongside it
            out.append(self._decode(cur, steps[i - 1]))
            cur = nxt
        out.append(self._decode(cur, steps[-1]))
        if self.cuda:
            torch.cuda.current_stream().wait_stream(self.dec_stream)
        return out

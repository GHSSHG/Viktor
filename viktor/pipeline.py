"""Encoder/decoder pipelining across batches on one GPU.

While batch i is decoded on the (high-priority) decoder stream, batch i+1 is encoded on the
encoder stream. The encoder is compute-bound and the decode steps are mostly KV-read-bound, so the
two can partly overlap. overlap=False runs the same work strictly in sequence for comparison.
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

    def _stream(self, s):
        return torch.cuda.stream(s) if self.cuda else contextlib.nullcontext()

    @torch.no_grad()
    def _encode(self, signal):
        if not self.cuda:
            return self.model.encoder(signal), None
        self.enc_stream.wait_stream(torch.cuda.current_stream())
        with self._stream(self.enc_stream):
            frames = self.model.encoder(signal)
            ev = torch.cuda.Event()
            ev.record()
        return frames, ev

    @torch.no_grad()
    def _decode(self, item, steps):
        frames, ev = item
        if not self.cuda:
            return self.engine.generate(frames, steps).to("cpu", copy=True)
        with self._stream(self.dec_stream):
            self.dec_stream.wait_event(ev)
            frames.record_stream(self.dec_stream)
            return self.engine.generate(frames, steps).to("cpu", copy=True)  # synchronises the decoder stream only

    @torch.no_grad()
    def run(self, signals, steps=None, overlap=True):
        """signals: iterable of [B, 1, chunksize] tensors on the device. Returns a list of token tensors."""
        out = []
        if not overlap:
            for sig in signals:
                out.append(self.engine.generate(self.model.encoder(sig), steps).to("cpu", copy=True))
            return out
        it = iter(signals)
        cur = self._encode(next(it))
        for sig in it:
            nxt = self._encode(sig)  # queued before decode(cur) blocks, so it runs alongside it
            out.append(self._decode(cur, steps))
            cur = nxt
        out.append(self._decode(cur, steps))
        if self.cuda:
            torch.cuda.current_stream().wait_stream(self.dec_stream)
        return out

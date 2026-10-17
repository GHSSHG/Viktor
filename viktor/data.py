"""Training windows (scripts/make_windows.py index) -> signal, label and per-base weights.

The signal is read from pod5 on the fly and preprocessed exactly as at inference (chunking.dorado_scale on
the child read, then the window x[win : win + 12288]); the label is fetched from the reference.
WindowDataset streams one split: each (rank, worker) owns a disjoint share of the index files (one per pod5
file), loads a few reads at a time (all their windows) and yields from a shuffle buffer.
"""

import glob
import os
import random

import numpy as np
import polars as pl
import pod5
import pysam
import torch

from .chunking import dorado_scale

SIZE = 12288
EDGE_WEIGHT = 0.5  # first and last base of a window
REF = os.path.expanduser("~/Work/BasecallerData/reference/hg002v1.2.fa")
_COMP = str.maketrans("ACGTN", "TGCAN")


def index_files(index_dir):
    return sorted(glob.glob(os.path.join(index_dir, "*.pod5.parquet")))


def read_signals(reader, rows, mean, stdev):
    """rows: index rows (dicts) whose parent reads are in the open pod5 reader -> standardised fp16 windows."""
    recs = {}
    for rec in reader.reads(selection=list({r["parent_id"] for r in rows})):
        recs[str(rec.read_id)] = (rec.signal, rec.calibration.offset, rec.calibration.scale,
                                  getattr(rec, "open_pore_level", float("nan")), rec.run_info.flow_cell_product_code)
    out = []
    for r in rows:
        raw, offset, scale, pore, flow_cell = recs[r["parent_id"]]
        child = raw[r["split_start"]: r["split_start"] + r["num_samples"]]
        out.append(dorado_scale(child, offset, scale, mean, stdev, pore, flow_cell)[r["win"]: r["win"] + SIZE])
    return out


def label(fa, r):
    s = fa.fetch(r["contig"], r["g_lo"], r["g_hi"]).upper()
    return s.translate(_COMP)[::-1] if r["strand"] == "-" else s


def base_weights(r, seq):
    """Read weight per base; masked and N bases 0; first and last base x EDGE_WEIGHT."""
    w = np.full(len(seq), r["w"], np.float32)
    for lo, hi in zip(r["mask_lo"], r["mask_hi"]):
        w[lo:hi] = 0
    w[np.frombuffer(seq.encode(), np.uint8) == ord("N")] = 0
    w[[0, -1]] *= EDGE_WEIGHT
    return w


def load_windows(rows, mean, stdev, ref=REF):
    """Index rows (any pod5 files) -> list of (signal, label, weights), in row order."""
    fa = pysam.FastaFile(ref)
    out = [None] * len(rows)
    by_file = {}
    for i, r in enumerate(rows):
        by_file.setdefault(r["pod5"], []).append(i)
    for path, ids in by_file.items():
        with pod5.Reader(path) as reader:
            for i, x in zip(ids, read_signals(reader, [rows[i] for i in ids], mean, stdev)):
                seq = label(fa, rows[i])
                out[i] = (x, seq, base_weights(rows[i], seq))
    return out


class WindowDataset(torch.utils.data.IterableDataset):
    """Yields (signal fp16 [12288], label str, weights float32 [len]) forever, reshuffled every pass."""

    def __init__(self, index_dir, split, mean, stdev, rank=0, world=1, seed=0, ref=REF,
                 reads_per_load=32, buffer=4096):
        self.files, self.split, self.ref = index_files(index_dir), split, ref
        self.mean, self.stdev, self.rank, self.world, self.seed = mean, stdev, rank, world, seed
        self.reads_per_load, self.buffer = reads_per_load, buffer

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        nw, wid = (wi.num_workers, wi.id) if wi else (1, 0)
        gid, G = self.rank * nw + wid, self.world * nw
        mine = [f for i, f in enumerate(self.files) if i % G == gid] or [self.files[gid % len(self.files)]]
        rng = random.Random(self.seed * 100003 + gid)
        fa = pysam.FastaFile(self.ref)
        buf = []
        while True:
            for path in rng.sample(mine, len(mine)):
                idx = pl.read_parquet(path).filter(pl.col("split") == self.split)
                if idx.height == 0:
                    continue
                groups = [g.to_dicts() for g in idx.partition_by("read_id")]
                rng.shuffle(groups)
                with pod5.Reader(groups[0][0]["pod5"]) as reader:
                    for i in range(0, len(groups), self.reads_per_load):
                        rows = [r for g in groups[i:i + self.reads_per_load] for r in g]
                        for r, x in zip(rows, read_signals(reader, rows, self.mean, self.stdev)):
                            seq = label(fa, r)
                            buf.append((x, seq, base_weights(r, seq)))
                        while len(buf) >= self.buffer:
                            yield buf.pop(rng.randrange(len(buf)))


def collate(items):
    sig = torch.from_numpy(np.stack([x for x, _, _ in items]))[:, None]
    return sig, [s for _, s, _ in items], [w for _, _, w in items]

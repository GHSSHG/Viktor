"""Training windows (scripts/make_windows.py index) -> signal, label and per-base weights.

The signal is read from pod5 on the fly and preprocessed exactly as at inference (chunking.dorado_scale on
the child read, then the window x[win : win + 12288]); the label is fetched from the reference.
WindowDataset streams one split: each (rank, worker) owns a disjoint share of the index files (one per pod5
file), loads a few reads at a time (all their windows) and yields from a shuffle buffer.
Training windows also carry each label base's prefix frame (window-local encoder frame of its signal midpoint),
from the same stage-6 forced alignment make_windows.py used (stage-6 file of the same name), for the attention
alignment loss.
Training and validation use only windows of full-quality reads (read weight w == 1); the chr20 'heldout'
test set keeps every read, so it stays the same for every model compared on it.
"""

import glob
import os
import random

import numpy as np
import polars as pl
import pod5
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pysam
import torch

from .chunking import DNA_TRIM_START, dorado_scale

SIZE = 12288
STRIDE = 12  # signal samples per encoder frame
CRF_FRAME = 6  # signal samples per stage-6 (CRF) frame
EDGE_WEIGHT = 0.5  # first and last base of a window
REF = os.path.expanduser("~/Work/BasecallerData/reference/hg002v1.2.fa")
STAGE6 = os.path.expanduser("~/Work/BasecallerData/stage6_align")
SEGMENTS = os.path.expanduser("~/Work/BasecallerData/inventory/truth_segments_nocigar.parquet")
_COMP = str.maketrans("ACGTN", "TGCAN")


def index_files(index_dir):
    return sorted(glob.glob(os.path.join(index_dir, "*.pod5.parquet")))


def split_rows(index, split):
    """Index rows (DataFrame or LazyFrame) of one split; low-quality reads (w < 1) are excluded except from
    the shared 'heldout' test set."""
    keep = pl.col("split") == split
    return index.filter(keep if split == "heldout" else keep & (pl.col("w") == 1))


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


def load_segments(stage6_file, read_ids, segments=SEGMENTS):
    """Stage-6 alignments ('ok' segments) of the given reads: flat uint16 dwell array and {read_id: [(offset, n,
    f_first, f_exit, contig, strand, t_start, t_end), ...]} (dwell of a segment = flat[offset : offset + n - 1])."""
    ids = pa.array(list(read_ids), pa.string())
    t = pq.read_table(stage6_file, columns=["read_id", "seg", "n", "f_first", "f_exit", "dwell", "status"])
    t = t.filter(pc.and_(pc.equal(t["status"], "ok"), pc.is_in(t["read_id"], ids)))
    dwell = t["dwell"].combine_chunks()
    flat, offs = dwell.values.to_numpy(zero_copy_only=False), dwell.offsets.to_numpy()
    ref = {(r[0], r[1]): r[2:] for r in pl.scan_parquet(segments)
           .select("read_id", "seg", "contig", "strand", "t_start", "t_end")
           .filter(pl.col("read_id").is_in(pl.Series(ids.to_pylist()))).collect().iter_rows()}
    segs = {}
    for i, (rid, sg, n, f0, fx) in enumerate(zip(*(t[c].to_pylist() for c in ("read_id", "seg", "n", "f_first", "f_exit")))):
        if (rid, sg) in ref:
            segs.setdefault(rid, []).append((int(offs[i]), n, f0, fx, *ref[rid, sg]))
    return flat, segs


def base_centers(r, flat, segs):
    """Window-local encoder frame of each label base's signal midpoint (read order), as make_windows.window_rows
    derives the label: base i enters at CRF frame e_i = f_first + cumsum(dwell), midpoint 3 * (e_i + e_{i+1}).
    The segment is the one holding the window in both signal and reference (segments of fold-back reads overlap
    in signal)."""
    A = r["win"] + DNA_TRIM_START  # window start in child-read samples
    for off, n, f0, fx, contig, strand, t0, t1 in segs[r["read_id"]]:
        if (CRF_FRAME * f0 <= A and A + SIZE <= CRF_FRAME * (fx + 1) and contig == r["contig"]
                and strand == r["strand"] and t0 <= r["g_lo"] and r["g_hi"] <= t1):
            e = np.empty(n + 1, np.int64)
            e[0], e[n] = f0, fx + 1
            e[1:n] = f0 + np.cumsum(flat[off:off + n - 1], dtype=np.int64)
            mid = CRF_FRAME // 2 * (e[:-1] + e[1:])
            lo, hi = np.searchsorted(mid, A), np.searchsorted(mid, A + SIZE)
            assert hi - lo == r["n_bases"], (r["read_id"], r["win"])
            return ((mid[lo:hi] - A - STRIDE / 2) / STRIDE).astype(np.float32)
    raise KeyError(f"no stage-6 segment holds window {r['read_id']} {r['win']}")


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
    """Yields (signal fp16 [12288], label str, weights float32 [len], base frames float32 [len]) forever,
    reshuffled every pass."""

    def __init__(self, index_dir, split, mean, stdev, rank=0, world=1, seed=0, ref=REF, stage6=STAGE6,
                 reads_per_load=32, buffer=4096):
        self.files, self.split, self.ref, self.stage6 = index_files(index_dir), split, ref, stage6
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
                idx = split_rows(pl.read_parquet(path), self.split)
                if idx.height == 0:
                    continue
                groups = [g.to_dicts() for g in idx.partition_by("read_id")]
                rng.shuffle(groups)
                flat, segs = load_segments(os.path.join(self.stage6, os.path.basename(path)), idx["read_id"].unique())
                with pod5.Reader(groups[0][0]["pod5"]) as reader:
                    for i in range(0, len(groups), self.reads_per_load):
                        rows = [r for g in groups[i:i + self.reads_per_load] for r in g]
                        for r, x in zip(rows, read_signals(reader, rows, self.mean, self.stdev)):
                            seq = label(fa, r)
                            buf.append((x, seq, base_weights(r, seq), base_centers(r, flat, segs)))
                        while len(buf) >= self.buffer:
                            yield buf.pop(rng.randrange(len(buf)))


def collate(items):
    """(signal [B, 1, 12288], labels, weights[, base frames]) from WindowDataset or load_windows items."""
    sig = torch.from_numpy(np.stack([it[0] for it in items]))[:, None]
    return (sig, *([it[i] for it in items] for i in range(1, len(items[0]))))

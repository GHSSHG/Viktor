"""Training windows (scripts/make_windows.py / make_windows_v3.py index) -> signal, label and per-base weights.

The signal is read from pod5 on the fly and preprocessed exactly as at inference (chunking.dorado_scale on
the child read, then the window x[win : win + win_len], win_len 12288 except a read's short last chunk); collate
repeat-pads a short window to the encoder granularity as at inference (Basecaller.encode drops the padded frames).
The label is fetched from the reference (v3 read-start windows: ADAPTER[ad_i0:ad_i1] + reference), unless the row
carries its own (label column: resolved v3 windows, possibly empty = the window holds no DNA).
WindowDataset streams one split: each (rank, worker) owns a disjoint share of the index files (one per pod5
file), loads a few reads at a time (all their windows) and yields from a shuffle buffer.
Training windows also carry each label base's prefix frame (window-local encoder frame of its signal midpoint),
from the same stage-6 forced alignment make_windows.py used, for the attention alignment loss: index rows carry
their segment (seg_off ...) and its dwell is memory-mapped from INDEX_DIR/dwell/<file>.npy; older indexes without it
read the stage-6 file of the same name (load_segments, several GB per worker while loading).
Training and validation use only windows of full-quality reads (read weight w == 1); the chr20 'heldout'
test set keeps every read, so it stays the same for every model compared on it.
PARTS lists the BasecallerData sources (HG002 parts 1-3, the Sowpati bacteria, the Hall et al. 2024 bacteria); MixedDataset draws
from several of them with fixed probabilities.
"""

import glob
import os
import random

import numpy as np
import polars as pl
import pod5
import pyarrow as pa
import pyarrow.parquet as pq
import pysam
import torch

from .chunking import DNA_TRIM_START, dorado_scale, repeat_pad

SIZE = 12288
STRIDE = 12  # signal samples per encoder frame
GRANULARITY = STRIDE * 64  # short chunks are repeat-padded to a multiple of this (Dorado chunk_size_granularity)
ADAPTER = "CCTGTACTTCGTTCAGTTACGTATTGCT"  # ligation adapter of v3 read-start labels (make_windows_v3.py)
CRF_FRAME = 6  # signal samples per stage-6 (CRF) frame
EDGE_WEIGHT = 0.5  # first and last base of a window
BD = os.path.expanduser("~/Work/BasecallerData")
# source -> BasecallerData table prefix, stage-6 dir, reference, pod5 glob, chimera mask table (same dirs as
# BasecallerData/scripts/stage6_forced_align.py)
PARTS = {
    "part1": dict(prefix="", stage6="stage6_align", ref="reference/hg002v1.2.fa",
                  pod5="/data/hg002_giab2025.01/pod5/*/*.pod5", chimera="chimera_bp_v2"),
    "part2": dict(prefix="part2_", stage6="stage6_part2", ref="reference/hg002v1.2.fa",
                  pod5="/data/hg002_giab2025.01_more/pod5/*/*.pod5", chimera="part2_chimera_bp"),
    "part3": dict(prefix="part3_", stage6="stage6_part3", ref="reference/hg002v1.2.fa",
                  pod5="/data/hg002_giab2025.01_rest/pod5/*/*.pod5", chimera="part3_chimera_bp"),
    "bact": dict(prefix="bact_", stage6="stage6_bact", ref="reference/bacteria_sft3.fa",
                 pod5="/data_intel/sowpati_basemod/Raw/pod5/bacteria/*/*.pod5", chimera="bact_chimera_bp"),
    "hall": dict(prefix="hall_", stage6="stage6_hall", ref="reference/bacteria_hall.fa",
                 pod5="/data_intel/bacteria_hall2024/pod5/*/*.pod5", chimera="hall_chimera_bp"),
}


def part_paths(part):
    """(reference FASTA, stage-6 dir, truth segment table) of a source."""
    p = PARTS[part]
    return (os.path.join(BD, p["ref"]), os.path.join(BD, p["stage6"]),
            os.path.join(BD, "inventory", p["prefix"] + "truth_segments_nocigar.parquet"))


REF, STAGE6, SEGMENTS = part_paths("part1")
_COMP = str.maketrans("ACGTN", "TGCAN")


def index_files(index_dir):
    return sorted(glob.glob(os.path.join(index_dir, "*.pod5.parquet")))


def pod5_path(path):
    """The index's pod5 path, or its copy under /data_nvme when /data is unavailable (HG002 part 2 is mirrored there)."""
    alt = path.replace("/data/", "/data_nvme/", 1)
    return alt if path.startswith("/data/") and not os.path.exists(path) and os.path.exists(alt) else path


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
        # copy: a view would keep the whole scaled read alive (the shuffle buffer holds thousands of windows)
        x = dorado_scale(child, offset, scale, mean, stdev, pore, flow_cell)
        out.append(x[r["win"]: r["win"] + win_len(r)].copy())
    return out


def win_len(r):
    return r.get("win_len") or SIZE


def _own(r, k):
    """Per-row weights / centers of a head window (make_head_labels.py); None for reference-labelled rows."""
    v = r.get(k)
    return v if v is not None and len(v) else None


def label(fa, r):
    if r.get("label") is not None:  # own label (head / resolved windows; "" = no DNA in the window)
        return r["label"]
    s = fa.fetch(r["contig"], r["g_lo"], r["g_hi"]).upper()
    s = s.translate(_COMP)[::-1] if r["strand"] == "-" else s
    return ADAPTER[r["ad_i0"]:r["ad_i1"]] + s if r.get("ad_i1") else s


def base_weights(r, seq):
    """Read weight per base; masked and N bases 0; first and last base x EDGE_WEIGHT."""
    if _own(r, "lw") is not None:  # head window: the weights are stored (0 for bases of unknown identity)
        return np.asarray(r["lw"], np.float32) * r["w"]
    w = np.full(len(seq), r["w"], np.float32)
    for lo, hi in zip(r["mask_lo"], r["mask_hi"]):
        w[lo:hi] = 0
    w[np.frombuffer(seq.encode(), np.uint8) == ord("N")] = 0
    if len(w):
        w[[0, -1]] *= EDGE_WEIGHT
    return w


def load_segments(stage6_file, read_ids, segments=SEGMENTS):
    """Stage-6 alignments ('ok' segments) of the given reads: flat uint16 dwell array and {read_id: [(offset, n,
    f_first, f_exit, contig, strand, t_start, t_end), ...]} (dwell of a segment = flat[offset : offset + n - 1])."""
    ids = list(read_ids)
    t = pq.read_table(stage6_file, columns=["read_id", "seg", "n", "f_first", "f_exit", "dwell", "status"])
    dwell = t["dwell"].combine_chunks()  # the whole file, unfiltered: no copy of the ~2 GB dwell column
    flat, offs = dwell.values.to_numpy(zero_copy_only=False), dwell.offsets.to_numpy()
    ref = {(r[0], r[1]): r[2:] for r in pl.scan_parquet(segments)
           .select("read_id", "seg", "contig", "strand", "t_start", "t_end")
           .filter(pl.col("read_id").is_in(ids)).collect().iter_rows()}
    segs = {}
    cols = (t[c].to_pylist() for c in ("read_id", "seg", "n", "f_first", "f_exit", "status"))
    for i, (rid, sg, n, f0, fx, status) in enumerate(zip(*cols)):
        if status == "ok" and (rid, sg) in ref:
            segs.setdefault(rid, []).append((int(offs[i]), n, f0, fx, *ref[rid, sg]))
    return flat, segs


def _centers(r, flat, off, n, f0, fx):
    A = r["win"] + DNA_TRIM_START  # window start in child-read samples
    e = np.empty(n + 1, np.int64)
    e[0], e[n] = f0, fx + 1
    e[1:n] = f0 + np.cumsum(flat[off:off + n - 1], dtype=np.int64)
    mid = CRF_FRAME // 2 * (e[:-1] + e[1:])
    lo, hi = np.searchsorted(mid, A), np.searchsorted(mid, A + win_len(r))
    mid = mid[lo:hi]
    if r.get("ad_i1"):  # v3 read start: its adapter bases (entry frames ad_e; the last one exits at f_first) come first
        ea = np.append(np.asarray(r["ad_e"], np.int64), e[0])
        mid = np.concatenate([(CRF_FRAME // 2 * (ea[:-1] + ea[1:]))[r["ad_i0"]:r["ad_i1"]], mid])
    assert len(mid) == r["n_bases"], (r["read_id"], r["win"])
    return ((mid - A - STRIDE / 2) / STRIDE).astype(np.float32)


def base_centers(r, flat, segs=None):
    """Window-local encoder frame of each label base's signal midpoint (read order), as make_windows.window_rows
    derives the label: base i enters at CRF frame e_i = f_first + cumsum(dwell), midpoint 3 * (e_i + e_{i+1}).
    segs None: the row names its segment (flat = the index's dwell/<file>.npy). Otherwise (load_segments) the
    segment is the one holding the window in both signal and reference (segments of fold-back reads overlap in
    signal)."""
    if _own(r, "lc") is not None:
        return np.asarray(r["lc"], np.float32)
    if r.get("seg_off") is None and segs is None:  # resolved v3 windows: no alignment target
        return np.full(r["n_bases"], -1.0, np.float32)
    if segs is None:
        return _centers(r, flat, r["seg_off"], r["seg_n"], r["f_first"], r["f_exit"])
    A = r["win"] + DNA_TRIM_START
    for off, n, f0, fx, contig, strand, t0, t1 in segs[r["read_id"]]:
        if (CRF_FRAME * f0 <= A and A + SIZE <= CRF_FRAME * (fx + 1) and contig == r["contig"]
                and strand == r["strand"] and t0 <= r["g_lo"] and r["g_hi"] <= t1):
            return _centers(r, flat, off, n, f0, fx)
    raise KeyError(f"no stage-6 segment holds window {r['read_id']} {r['win']}")


def load_windows(rows, mean, stdev, ref=REF):
    """Index rows (any pod5 files) -> list of (signal, label, weights), in row order."""
    fa = pysam.FastaFile(ref)
    out = [None] * len(rows)
    by_file = {}
    for i, r in enumerate(rows):
        by_file.setdefault(r["pod5"], []).append(i)
    for path, ids in by_file.items():
        with pod5.Reader(pod5_path(path)) as reader:
            for i, x in zip(ids, read_signals(reader, [rows[i] for i in ids], mean, stdev)):
                seq = label(fa, rows[i])
                out[i] = (x, seq, base_weights(rows[i], seq))
    return out


class WindowDataset(torch.utils.data.IterableDataset):
    """Yields (signal fp16 [12288], label str, weights float32 [len], base frames float32 [len]) forever,
    reshuffled every pass."""

    def __init__(self, index_dir, split, mean, stdev, rank=0, world=1, seed=0, ref=REF, stage6=STAGE6,
                 segments=SEGMENTS, reads_per_load=32, buffer=4096):
        self.files, self.split, self.ref, self.stage6 = index_files(index_dir), split, ref, stage6
        self.segments = segments
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
                # rows stay columnar; only the reads being loaded become dicts (a whole file of dicts took GBs per worker)
                groups = (idx.with_row_index("_i").group_by("read_id", maintain_order=True)
                          .agg("_i")["_i"].to_list())  # row indices per read, in first-appearance order
                rng.shuffle(groups)
                flat = segs = None
                cache = os.path.join(os.path.dirname(path), "dwell", os.path.basename(path)[:-len(".parquet")] + ".npy")
                if os.path.exists(cache):
                    flat = np.load(cache, mmap_mode="r")
                elif "label" in idx.columns and "contig" not in idx.columns:
                    pass  # resolved v3 windows: own labels, no alignment
                else:
                    flat, segs = load_segments(os.path.join(self.stage6, os.path.basename(path)),
                                               idx["read_id"].unique(), self.segments)
                    pa.default_memory_pool().release_unused()  # return the previous file's and the read buffers
                with pod5.Reader(pod5_path(idx["pod5"][0])) as reader:
                    for i in range(0, len(groups), self.reads_per_load):
                        rows = idx[[j for g in groups[i:i + self.reads_per_load] for j in g]].to_dicts()
                        for r, x in zip(rows, read_signals(reader, rows, self.mean, self.stdev)):
                            seq = label(fa, r)
                            try:
                                centers = base_centers(r, flat, segs)
                                assert len(centers) == len(seq), "label / alignment length"
                            except (AssertionError, KeyError) as e:  # inconsistent alignment: skip the window
                                print(f"skip window {r['read_id']} {r['win']}: {e!r}", flush=True)
                                continue
                            buf.append((x, seq, base_weights(r, seq), centers))
                        while len(buf) >= self.buffer:
                            yield buf.pop(rng.randrange(len(buf)))


class MixedDataset(torch.utils.data.IterableDataset):
    """Yields WindowDataset items, each drawn from source i with probability weights[i] (every source streams
    on its own, with its own shuffle buffer)."""

    def __init__(self, datasets, weights, seed=0):
        self.datasets, self.weights, self.seed = datasets, weights, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 100003 + 7919 * (wi.id if wi else 0) + self.datasets[0].rank)
        its = [iter(d) for d in self.datasets]
        while True:
            yield next(rng.choices(its, self.weights)[0])


def collate(items):
    """(signal [B, 1, 12288], real samples per window [B], labels, weights[, base frames]) from WindowDataset or
    load_windows items. A short window is repeat-padded to the granularity (as at inference), then zero-padded."""
    sig = np.zeros((len(items), 1, SIZE), np.float16)
    for b, it in enumerate(items):
        x = repeat_pad(it[0], GRANULARITY)
        sig[b, 0, :len(x)] = x
    valid = torch.tensor([len(it[0]) for it in items])
    return (torch.from_numpy(sig), valid, *([it[i] for it in items] for i in range(1, len(items[0]))))

"""Which index rows (windows) a training run has drawn, replayed from its seed: no signals, no GPU.

train.py's stream is deterministic: per (rank, data worker) WindowDataset / MixedDataset RNGs seeded by (--seed * 1000 + start step), the
DataLoader hands worker w the batches k = w, w + workers, ... of a rank, so after N steps worker w has consumed the first
batch x ceil((N - w) / workers) windows of its own stream (the prefetched ones beyond that were never trained on). The replay runs the
real WindowDataset / MixedDataset code with the signal / label / alignment loading stubbed (no run so far skipped an inconsistent
window, so those never change the stream) and records the row of the index file (before the split filter) of every window.

  replay: python -m scripts.used_windows replay --windows DIR[:PART[:WEIGHT]] ... --seed 0 --start 0 --steps 250000 --world 3 --batch 42 --out runs/used/k4sft1_a
  merge:  python -m scripts.used_windows merge --out runs/used/all runs/used/k4sft1_a runs/used/k4sft1_b ...
  check:  python -m scripts.used_windows check --windows ... --seed S --start N --world W --batch B   (first batches of the real DataLoader vs replay)
Output DIR/<index dir name>/<index file>.npy: sorted unique rows (uint32); train.py --exclude-used DIR never draws them.
"""

import argparse
import collections
import contextlib
import glob
import itertools
import math
import multiprocessing as mp
import os
import types

import numpy as np
import polars as pl
import torch

import viktor.data as D
from viktor.data import part_paths


def parse_source(spec):
    """DIR[:PART[:WEIGHT]] -> (index dir, part, weight), as train.py."""
    d, part, w = (spec.split(":") + ["", ""])[:3]
    return d, part or "part1", float(w or 1)


def stub(real_labels):
    """Replace what loads signals (and, unless real_labels, labels / alignments) by row identities."""
    if getattr(D, "_stubbed", False):
        return
    D._stubbed = True
    pl_read = pl.read_parquet

    def read_parquet(path, *a, **k):
        df = pl_read(path, *a, **k)
        if not str(path).endswith(".pod5.parquet"):
            return df
        return df.with_row_index("_orow").with_columns(pl.lit(os.path.abspath(path)).alias("_file"))  # row = position in the file as stored

    pl.read_parquet = read_parquet
    D.read_signals = lambda reader, rows, mean, stdev: [(r["_file"], r["_orow"]) for r in rows]
    D.pod5 = types.SimpleNamespace(Reader=lambda p: contextlib.nullcontext())
    if not real_labels:
        D.label = lambda fa, r: ""
        D.base_centers = lambda r, flat, segs: []
        D.base_weights = lambda r, seq: None
        D.load_segments = lambda *a, **k: (None, None)


def dataset(sources, rank, world, seed):
    parts = [D.WindowDataset(d, "train", 0.0, 1.0, rank, world, seed, *part_paths(part)) for d, part, _ in sources]
    return parts[0] if len(parts) == 1 else D.MixedDataset(parts, [w for _, _, w in sources], seed)


def worker_stream(sources, rank, world, wid, nw, seed, limit, real_labels=False):
    stub(real_labels)
    torch.utils.data.get_worker_info = lambda: types.SimpleNamespace(num_workers=nw, id=wid)
    return itertools.islice(iter(dataset(sources, rank, world, seed)), limit)


def replay_one(job):
    sources, rank, world, wid, nw, seed, limit, dirs = job
    used = collections.defaultdict(list)
    for x, *_ in worker_stream(sources, rank, world, wid, nw, seed, limit):
        used[x[0]].append(x[1])
    return {k: np.array(v, np.uint32) for k, v in used.items()}, dirs


def cmd_replay(a):
    sources = [parse_source(s) for s in a.windows]
    jobs = []
    for rank in range(a.world):
        for wid in range(a.workers):
            n_batches = math.ceil((a.steps - wid) / a.workers)
            jobs.append((sources, rank, a.world, wid, a.workers, a.seed * 1000 + a.start, n_batches * a.batch, None))
    used = collections.defaultdict(list)
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for res, _ in pool.imap_unordered(replay_one, jobs):
            for f, rows in res.items():
                used[f].append(rows)
    n = 0
    for f, parts in used.items():
        src = os.path.basename(os.path.dirname(f))  # the index dir name (v3h and v3x share file names)
        os.makedirs(os.path.join(a.out, src), exist_ok=True)
        rows = np.unique(np.concatenate(parts))
        np.save(os.path.join(a.out, src, os.path.basename(f)[:-len(".parquet")] + ".npy"), rows)
        n += len(rows)
    print(f"{a.out}: {n} distinct windows (steps {a.start} -> {a.start + a.steps}, seed {a.seed}, world {a.world}, batch {a.batch})", flush=True)


def cmd_merge(a):
    files = collections.defaultdict(list)
    for seg in a.segments:
        for p in glob.glob(os.path.join(seg, "*", "*.npy")):
            files[os.path.relpath(p, seg)].append(p)
    n = 0
    for rel, ps in files.items():
        os.makedirs(os.path.dirname(os.path.join(a.out, rel)), exist_ok=True)
        rows = functools_union(ps)
        np.save(os.path.join(a.out, rel), rows)
        n += len(rows)
    print(f"{a.out}: {n} distinct windows in {len(files)} files from {len(a.segments)} segments", flush=True)


def cmd_weights(a):
    """Source weights for training on what is left: human (part1 / part2) sources share 1 - bact_share, the others bact_share, each split by
    its remaining train windows (train windows with w == 1 that --exclude does not list). Prints DIR:PART:WEIGHT ... for --windows."""
    sources = [parse_source(s) for s in a.windows]
    left = []
    for d, part, _ in sources:
        n = 0
        for f in glob.glob(os.path.join(d, "*.pod5.parquet")):
            df = pl.read_parquet(f, columns=["split", "w"]).with_row_index("_o")
            rows = df.filter((pl.col("split") == "train") & (pl.col("w") == 1))["_o"].to_numpy()
            skip = os.path.join(a.exclude, os.path.basename(os.path.normpath(d)), os.path.basename(f)[:-len(".parquet")] + ".npy")
            n += len(rows) - (np.isin(rows, np.load(skip)).sum() if os.path.exists(skip) else 0)
        left.append(n)
    human = [part in ("part1", "part2") for _, part, _ in sources]
    h, b = sum(n for n, x in zip(left, human) if x), sum(n for n, x in zip(left, human) if not x)
    print(" ".join(f"{d}:{part}:{(1 - a.bact_share) * n / h if x else a.bact_share * n / b:.4f}"
                   for (d, part, _), n, x in zip(sources, left, human)))
    print("remaining windows:", dict(zip((os.path.basename(os.path.normpath(d)) for d, _, _ in sources), left)), file=__import__("sys").stderr)


def functools_union(ps):
    return np.unique(np.concatenate([np.load(p) for p in ps])).astype(np.uint32)


def cmd_check(a):
    """The first --check-steps batches of rank 0, as the real DataLoader yields them (labels), against the replay (real labels)."""
    sources = [parse_source(s) for s in a.windows]
    ds = dataset(sources, 0, a.world, a.seed * 1000 + a.start)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, collate_fn=D.collate, num_workers=a.workers, prefetch_factor=4,
                                     multiprocessing_context="spawn")
    real = []
    for i, (sig, valid, seqs, *_) in enumerate(dl):
        real.append(list(seqs))
        if i + 1 >= a.check_steps:
            break
    ok = True
    for w in range(a.workers):
        gen = worker_stream(sources, 0, a.world, w, a.workers, a.seed * 1000 + a.start, a.batch * a.check_steps, True)
        items = [it[1] for it in gen]
        for k in range(w, a.check_steps, a.workers):
            mine = items[(k // a.workers) * a.batch:(k // a.workers + 1) * a.batch]
            same = mine == real[k]
            ok &= same
            print(f"step {k} (worker {w}): {'same' if same else 'DIFFERENT'}", flush=True)
    print("CHECK", "OK" if ok else "FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("replay", "check"):
        p = sub.add_parser(name)
        p.add_argument("--windows", nargs="+", required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--start", type=int, required=True, help="step the run (re)started at")
        p.add_argument("--world", type=int, required=True)
        p.add_argument("--batch", type=int, required=True, help="windows per GPU")
        p.add_argument("--workers", type=int, default=8)
        if name == "replay":
            p.add_argument("--steps", type=int, required=True, help="steps trained in this segment")
            p.add_argument("--out", required=True)
            p.add_argument("--procs", type=int, default=32)
        else:
            p.add_argument("--check-steps", type=int, default=16)
    w = sub.add_parser("weights")
    w.add_argument("--windows", nargs="+", required=True)
    w.add_argument("--exclude", required=True)
    w.add_argument("--bact-share", type=float, default=0.15)
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.add_argument("segments", nargs="+")
    a = ap.parse_args()
    {"replay": cmd_replay, "merge": cmd_merge, "check": cmd_check, "weights": cmd_weights}[a.cmd](a)

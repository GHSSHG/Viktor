"""Resolve the windows window cleaning v3 could not label (CLEANING_RULES.md, revision 2026-10-28, item 4) from stage-6 data alone.

make_windows_v3 keeps a window only if it lies inside ONE aligned segment (or past the end by <= END_TOL). Everything else goes to
<file>.unresolved.parquet (kinds junction / tail / unaligned). This pass labels the ones whose signal IS covered by aligned bases, no new alignment needed:

  joined    the window touches one or more stage-6 segments and the samples NOT covered by any aligned base (gaps between two segments, a few
            samples beside a segment edge) add up to <= --tol samples (default 120 = ~10 bases). Chimeric junctions are the main case: segment A
            ends where segment B begins (median gap 6 samples), so the true sequence is the end of A followed by the start of B in read order.
            label = the bases of every touching segment whose signal midpoint falls in the window, concatenated in read order.
  eos_flat  the window lies entirely past the end of the last segment and its signal is flat (median 120-sample block std < --flat x that of
            the aligned part of the read): open-pore / stalled signal. label = "" (immediately the stop symbol), hard bit 8.

Everything else stays unresolved (written to <file>.still_unresolved.parquet with the reason and the uncovered sample count):
  overlap   two segments overlap in signal by > tol samples (the same signal would be labelled by both: duplicated bases); small overlaps (<= tol,
            boundary jitter, median 42 samples) are accepted
  gap       > tol uncovered samples between / beside segments (sequence there unknown: needs forced alignment of the reference between the flanking
            alignments (colinear gaps only) or masking, a separate decision)
  tail_active  entirely past the last segment with non-flat signal
  start     before the first segment (read start without an adapter label)

Output <out>/<pod5 file>.resolved.parquet, one row per window: read_id parent_id pod5 split_start num_samples win win_len rule n_bases label (read
orientation, upper case, from the reference) mask_lo mask_hi (label coordinates) hard (1 few_bases, 2 stall, 8 eos, 16 short) w split n_pieces
uncovered contigs segs. Does not touch the files of make_windows_v3.

Usage (from ~/Work/Viktor):  python -m scripts.resolve_windows_v3 --part part1 --win ~/Work/BasecallerData/viktor_win_v3 --out ~/Work/BasecallerData/viktor_win_v3_resolved [--limit N] [--workers 8]
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import polars as pl

from viktor.chunking import DNA_TRIM_START, chunk_read
from viktor.data import BD, PARTS

sys.path.insert(0, os.path.join(BD, "scripts"))
from placement import flag_misplaced  # noqa: E402

from scripts.make_windows import HOLDOUT, load_meta  # noqa: E402
from scripts.make_windows_v3 import (END_TOL, FRAME, MAX_DWELL, OVERLAP, SIZE, _COMP, _entries)  # noqa: E402

BLOCK = 120


def piece(e, r, A, B, ivs):
    """bases of one segment whose signal midpoint lies in [A, B): -> None or (lo, hi, g_lo, g_hi, masks (relative to the piece), max dwell)"""
    mid = 3 * (e[:-1] + e[1:])
    lo, hi = np.searchsorted(mid, A), np.searchsorted(mid, B)
    if hi <= lo:
        return None
    t0, t1, minus = r["t_start"], r["t_end"], r["strand"] == "-"
    g_lo, g_hi = (t1 - hi, t1 - lo) if minus else (t0 + lo, t0 + hi)
    mk = []
    for s, t in ivs:
        s, t = max(s, g_lo), min(t, g_hi)
        if s < t:
            mk.append(((g_hi - t) if minus else (s - g_lo), (g_hi - s) if minus else (t - g_lo)))
    return lo, hi, int(g_lo), int(g_hi), mk, int(np.diff(e)[lo:hi].max())


def flat_ratio(z, A, B, ref_sd):
    w = z[A:B]
    n = len(w) // BLOCK
    if n < 1 or ref_sd <= 0:
        return None
    return float(np.median(w[:n * BLOCK].reshape(n, BLOCK).std(1)) / ref_sd)


def process(job):
    path, win_dir, out, seg, masks, patch, pod5_paths, flag_part, ref, tol, flat = job
    import pod5
    import pysam
    t = time.time()
    name = os.path.basename(path)[:-len(".parquet")]
    ufile = os.path.join(win_dir, name + ".unresolved.parquet")
    u = pl.read_parquet(ufile).filter(pl.col("kind") != "head_pending")
    stats = {"unresolved_in": u.height, "joined": 0, "eos_flat": 0, "gap": 0, "overlap": 0, "tail_active": 0, "start": 0, "no_segments": 0}
    raw = pl.read_parquet(path, columns=["read_id", "seg", "n", "f_first", "f_exit", "dwell", "status", "deficit"])
    if flag_part:
        sq = seg.select("read_id", "seg", "q_start", "q_end", "qlen")
        bad = flag_misplaced(raw.drop("dwell"), sq, seg.select("read_id", "num_samples", "trim_start").unique("read_id")).select("read_id", "seg")
        raw = raw.join(bad, on=["read_id", "seg"], how="anti")
    s6 = raw.filter(pl.col("status") == "ok").drop("status", "deficit").join(seg, on=["read_id", "seg"]).sort("read_id", "seg")
    s6 = s6.filter(pl.col("read_id").is_in(u["read_id"].unique().implode()))
    by_seg = {}
    for m in masks.filter(pl.col("read_id").is_in(s6["read_id"].unique().implode())).iter_rows():
        by_seg.setdefault((m[0], m[1]), []).append((m[3], m[4]))
    patch_by_contig = {}
    for c, s, e_ in patch.iter_rows():
        patch_by_contig.setdefault(c, []).append((s, e_))
    fa = pysam.FastaFile(ref)
    segs_of = {}
    for r in s6.iter_rows(named=True):
        segs_of.setdefault(r["read_id"], []).append(r)
    wins_of = {}
    for rid, w, wl, k in u.iter_rows():
        wins_of.setdefault(rid, []).append((w, wl, k))
    rows, left = [], []
    need_flat = []                                         # (read, win index in rows list placeholder)
    keys = ("read_id", "parent_id", "pod5", "split_start", "num_samples", "win", "win_len", "rule", "n_bases", "label", "mask_lo", "mask_hi",
            "hard", "w", "split", "n_pieces", "uncovered", "contigs", "segs")
    cand_flat = []
    for rid, wl in wins_of.items():
        rs = segs_of.get(rid)
        if not rs:
            stats["no_segments"] += len(wl)
            left += [(rid, w, l, "no_segments", -1) for w, l, k in wl]
            continue
        ents = []
        for r in rs:
            e = _entries({"n": r["n"], "f_first": r["f_first"], "f_exit": r["f_exit"], "dwell": np.asarray(r["dwell"], np.int64)})
            ivs = list(by_seg.get((r["read_id"], r["seg"]), ())) + [(s, t_) for s, t_ in patch_by_contig.get(r["contig"], [])
                                                                    if s < r["t_end"] and t_ > r["t_start"]]
            ents.append((e, r, ivs))
        order = sorted(range(len(ents)), key=lambda i: ents[i][0][0])
        lo_all, hi_all = FRAME * min(ents[i][0][0] for i in order), FRAME * max(ents[i][0][-1] for i in order)
        r0 = rs[0]
        nchunks = len(chunk_read(r0["num_samples"] - DNA_TRIM_START, SIZE, OVERLAP))
        for w, L, kind in wl:
            A, B = w + DNA_TRIM_START, w + DNA_TRIM_START + L
            pcs = []
            for i in order:
                e, r, ivs = ents[i]
                p = piece(e, r, A, B, ivs)
                if p is not None:
                    pcs.append((i, p, FRAME * int(e[p[0]]), FRAME * int(e[p[1]])))
            if not pcs:
                if A >= hi_all:
                    cand_flat.append((rid, w, L, A, B, lo_all, hi_all, r0))
                elif B <= lo_all:
                    stats["start"] += 1; left.append((rid, w, L, "start", L))
                else:
                    stats["gap"] += 1; left.append((rid, w, L, "gap", L))
                continue
            cov = sum(min(b, B) - max(a, A) for _, _, a, b in pcs)
            unc = L - cov
            if unc > tol:
                stats["gap"] += 1; left.append((rid, w, L, "gap", unc))
                continue
            if unc < -tol:                                 # two segments claim the same signal: bases would be labelled twice
                stats["overlap"] += 1; left.append((rid, w, L, "overlap", unc))
                continue
            lab, mlo, mhi, nb, hard, off = [], [], [], 0, 0, 0
            for i, (lo, hi, g_lo, g_hi, mk, mx), _, _ in pcs:
                r = ents[i][1]
                s = fa.fetch(r["contig"], g_lo, g_hi).upper()
                lab.append(s.translate(_COMP)[::-1] if r["strand"] == "-" else s)
                mlo += [off + a for a, b in mk]; mhi += [off + b for a, b in mk]
                off += hi - lo
                hard |= 2 if mx > MAX_DWELL else 0
            nb = off
            hard |= (16 if L < SIZE else 0) | (1 if nb < L // 40 else 0)
            if w == chunk_read(r0["num_samples"] - DNA_TRIM_START, SIZE, OVERLAP)[-1][0] and B > hi_all and B - hi_all <= END_TOL:
                hard |= 8
            c0 = ents[pcs[0][0]][1]
            split = "heldout" if any(ents[i][1]["contig"].startswith(HOLDOUT) for i, *_ in pcs) else c0["split"]
            rows.append((rid, c0["parent_id"], pod5_paths[c0["pod5_file"]], c0["split_start"], c0["num_samples"], w, L, "joined", nb,
                         "".join(lab), mlo, mhi, hard, c0["w_channel"], split, len(pcs), int(unc),
                         [ents[i][1]["contig"] for i, *_ in pcs], [int(ents[i][1]["seg"]) for i, *_ in pcs]))
            stats["joined"] += 1
    if cand_flat:                                          # flat-signal test needs the raw signal
        sel = {}
        for c in cand_flat:
            sel.setdefault(c[7]["parent_id"], []).append(c)
        pth = pod5_paths[cand_flat[0][7]["pod5_file"]]
        with pod5.Reader(pth) as rd:
            for rec in rd.reads(selection=list(sel), missing_ok=True):
                cs = sel[str(rec.read_id)]
                r0 = cs[0][7]
                s = (rec.signal.astype(np.float32) + rec.calibration.offset) * rec.calibration.scale
                s = s[r0["split_start"]:r0["split_start"] + r0["num_samples"]][DNA_TRIM_START:]
                z = (s - np.median(s)) / (np.median(np.abs(s - np.median(s))) * 1.4826 + 1e-6)
                for rid, w, L, A, B, lo_all, hi_all, r in cs:
                    a0, b0 = max(lo_all - DNA_TRIM_START, 0), max(hi_all - DNA_TRIM_START, 0)
                    n = max(min((b0 - a0), len(z) - a0) // BLOCK, 0)
                    ref_sd = float(np.median(z[a0:a0 + n * BLOCK].reshape(n, BLOCK).std(1))) if n > 0 else 0.0
                    fr = flat_ratio(z, w, w + L, ref_sd)
                    if fr is not None and fr < flat:
                        rows.append((rid, r["parent_id"], pod5_paths[r["pod5_file"]], r["split_start"], r["num_samples"], w, L, "eos_flat", 0, "",
                                     [], [], 8 | (16 if L < SIZE else 0), r["w_channel"], "heldout" if r["contig"].startswith(HOLDOUT) else r["split"],
                                     0, int(L), [], []))
                        stats["eos_flat"] += 1
                    else:
                        stats["tail_active"] += 1; left.append((rid, w, L, "tail_active", L))
    sch = {"win": pl.Int32, "win_len": pl.Int32, "n_bases": pl.Int32, "hard": pl.Int8, "uncovered": pl.Int32, "n_pieces": pl.Int8, "split_start": pl.Int64,
           "num_samples": pl.Int64, "w": pl.Float32, "mask_lo": pl.List(pl.Int32), "mask_hi": pl.List(pl.Int32), "contigs": pl.List(pl.String),
           "segs": pl.List(pl.Int32)}
    pl.DataFrame(rows, schema=list(keys), orient="row", schema_overrides=sch).write_parquet(os.path.join(out, name + ".resolved.parquet"), compression="zstd")
    pl.DataFrame(left, schema=["read_id", "win", "win_len", "reason", "uncovered"], orient="row",
                 schema_overrides={"win": pl.Int32, "win_len": pl.Int32, "uncovered": pl.Int32}).write_parquet(os.path.join(out, name + ".still_unresolved.parquet"))
    return {"file": name, **stats, "seconds": round(time.time() - t, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="part1", choices=sorted(PARTS))
    ap.add_argument("--win", required=True, help="make_windows_v3 output dir (has *.unresolved.parquet)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage6", default=None)
    ap.add_argument("--tol", type=int, default=120, help="max uncovered samples in a joined window")
    ap.add_argument("--flat", type=float, default=0.25, help="eos_flat: median block std < this x that of the aligned part")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--files", default="")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seg, masks, patch = load_meta(a.part)
    pod5_paths = {os.path.basename(p): p for p in glob.glob(PARTS[a.part]["pod5"])}
    s6dir = a.stage6 or f"{BD}/{PARTS[a.part]['stage6']}"
    files = sorted(glob.glob(f"{s6dir}/*.parquet"))
    if a.files:
        k = set(a.files.split(","))
        files = [f for f in files if os.path.basename(f)[:-8] in k]
    files = [f for f in files if os.path.exists(os.path.join(a.win, os.path.basename(f)[:-8] + ".unresolved.parquet"))][:a.limit]
    ref = os.path.join(BD, PARTS[a.part]["ref"])
    jobs = []
    for f in files:
        pod = os.path.basename(f)[:-8]
        if os.path.exists(os.path.join(a.out, pod + ".resolved.parquet")):
            continue
        part = seg.filter(pl.col("pod5_file") == pod)
        jobs.append((f, a.win, a.out, part, masks.filter(pl.col("read_id").is_in(part["read_id"].implode())), patch, pod5_paths,
                     a.part != "part1", ref, a.tol, a.flat))
    print(f"{len(jobs)} files", flush=True)
    stats = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for s in pool.imap_unordered(process, jobs):
            stats.append(s)
            print(json.dumps(s), flush=True)
    if stats:
        tot = {k: sum(s[k] for s in stats) for k in stats[0] if k not in ("file", "seconds")}
        print(json.dumps(tot), flush=True)


if __name__ == "__main__":
    main()

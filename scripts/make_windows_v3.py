"""Window cleaning v3 (CLEANING_RULES.md, revision 2026-10-28): cut EVERY read exactly like Dorado slides its window and find the
base sequence that belongs to each window.

Chunking = viktor.chunking.chunk_read: 12288 samples (after the 10-sample DNA trim), step 12288 - 600 = 11688 (overlap 600 samples =
100 CRF frames), the remainder is its own SHORT last chunk; at load time every chunk is repeat-padded with its own start to a multiple
of 768 (chunking.repeat_pad). A read shorter than one chunk is one short chunk. Nothing is dropped for being a read start, a stall,
few bases or a short chunk.

Base sequence of a window = the bases whose signal midpoint falls inside the window, from the stage-6 CRF forced alignment of the
truth sequence (truth BAM: ADAPTER + reference[t_start:t_end]). The reference part comes from stage 6 (dwell), the 28 adapter bases from
the head pass (scripts/stage6_head.py in BasecallerData): <head dir> for most reads, <head dir>_full (whole-read alignment) for reads
whose head alignment was a problem. Read-start window: label = ADAPTER[ad_i0:ad_i1] + ref[g_lo:g_hi].

Windows that leave the aligned region are NOT silently dropped: they go to <file>.unresolved.parquet (read_id, win, win_len, kind) for the
whole-read correction pass. Kinds: head_pending (read start, the adapter has no head alignment yet), tail (past the aligned end by more
than END_TOL samples), junction (touches two segments), unaligned (partly outside a single segment: internal unaligned stretch, or a
read start without adapter). A last chunk that ends at most END_TOL samples after the aligned end is kept and flagged eos (the label ends
with the stop symbol).

Output (one parquet per pod5 file, one row per window; the old make_windows.py columns plus):
  win_len     samples of the chunk (12288, or less for the short last chunk; the signal is x[win : win + win_len], repeat-padded to 768)
  ad_i0 ad_i1 adapter bases of the label (ADAPTER[ad_i0:ad_i1], 0,0 = none); ad_e = entry frames (CRF frames, child read) of the 28 adapter bases
  hard        bit 1 few_bases, 2 stall (> 200 frames on one base), 4 adapter in window, 8 eos, 16 short chunk
  dwell/<file>.npy as before (dwell of the REFERENCE part of each segment that has windows)
--bam additionally writes <out>/bam/<file>.bam: unmapped BAM, one record per window, SEQ = the window's base sequence
(name <read_id>:<win>, tags ct ts te sd wn wl ad hd w).

Example:
  python -m scripts.make_windows_v3 build --part part1 --out ~/Work/BasecallerData/viktor_win_v3 --workers 24 --bam
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

SIZE, OVERLAP, FRAME = 12288, 600, 6
MAX_DWELL = 200
END_TOL = 1200          # samples between the aligned end and the read end that still count as "the read ends here" (eos)
SPLICE_TOL = 12         # frames, head pass vs stage 6 at reference base 0
ADAPTER = "CCTGTACTTCGTTCAGTTACGTATTGCT"
NA = len(ADAPTER)
MAX_GAP = 100           # adapter rule of make_truth_bam.py: first segment by q_start, q_start <= MAX_GAP, not split, trim_start > 0
HEAD_DIRS = {"part1": "stage6_head", "part2": "stage6_head_part2", "part3": "stage6_head_part3"}
_COMP = str.maketrans("ACGTN", "TGCAN")


def want_adapter(part):
    """DataFrame (read_id, seg, want_ad): the segments that carry the adapter in the truth BAM."""
    P = PARTS[part]["prefix"]
    t = pl.read_parquet(f"{BD}/inventory/{P}truth_segments_nocigar.parquet", columns=["read_id", "seg", "q_start"])
    b = pl.read_parquet(f"{BD}/inventory/{P}bam_reads.parquet", columns=["read_id", "is_split", "trim_start"])
    t = t.sort(["read_id", "q_start"]).group_by("read_id", maintain_order=True).first().join(b, on="read_id")
    return t.filter((pl.col("q_start") <= MAX_GAP) & ~pl.col("is_split") & (pl.col("trim_start") > 0)).select("read_id", "seg").with_columns(
        pl.lit(True).alias("want_ad"))


def _entries(row):
    n = row["n"]
    e = np.empty(n + 1, np.int64)
    e[0], e[n] = row["f_first"], row["f_exit"] + 1
    e[1:n] = row["f_first"] + np.cumsum(np.asarray(row["dwell"], np.int64))
    return e


def adapter_frames(read_id, seg, f_first, n_ref, head, full):
    """-> (e_ref (n_ref + 1 entries) or None to keep stage 6's, ea (NA adapter entries) or None). The whole-read alignment replaces
    the stage-6 frames of the segment; the head alignment only adds the adapter (checked against stage 6's f_first)."""
    fr = full.get((read_id, seg))
    if fr is not None and fr["n"] == NA + n_ref:
        e = _entries(fr)
        return e[NA:], e[:NA]
    hr = head.get((read_id, seg))
    if hr is not None:
        e = _entries(hr)
        if len(e) > NA and abs(e[NA] - f_first) <= SPLICE_TOL:
            return None, e[:NA]
    return None, None


def read_aligned(path):
    if not os.path.exists(path):
        return {}
    t = pl.read_parquet(path, columns=["read_id", "seg", "n", "f_first", "f_exit", "dwell", "status"]).filter(pl.col("status") == "ok")
    return {(r["read_id"], r["seg"]): r for r in t.iter_rows(named=True)}


def read_windows(r, e, ea, masks, patch, chunks, tail_ok):
    """Windows of one segment. r: segment row; e: entry frames of the reference bases (n + 1); ea: adapter entry frames or None.
    Returns {chunk index: row}."""
    n = r["n"]
    dw = np.diff(e)
    mid = 3 * (e[:-1] + e[1:])
    seg_lo, seg_hi = FRAME * e[0], FRAME * e[n]
    if ea is not None:
        eaa = np.append(ea, e[0])
        amid = 3 * (eaa[:-1] + eaa[1:])
        adw = np.diff(eaa)
    t0, t1, minus = r["t_start"], r["t_end"], r["strand"] == "-"
    ivs = list(masks) + [(s, t) for s, t in patch.get(r["contig"], []) if s < t1 and t > t0]
    out = {}
    for ci, (a, b) in enumerate(chunks):
        A, L = a + DNA_TRIM_START, b - a
        first, last = ci == 0, ci == len(chunks) - 1
        if ea is not None and first and r["want_ad"]:
            lo_ok = True                                  # lead-in before the adapter has no bases
        else:
            lo_ok = A >= seg_lo
        if not lo_ok:
            continue
        eos = 0
        if A + L > seg_hi:
            if last and tail_ok and A + L - seg_hi <= END_TOL and A < seg_hi:
                eos = 8
            else:
                continue
        if ea is not None and first and r["want_ad"]:
            i0, i1 = np.searchsorted(amid, A), np.searchsorted(amid, A + L)
            if A > seg_lo and i1 > i0:
                i0 = i1                                   # (cannot happen for the first chunk, kept for safety)
        else:
            i0 = i1 = 0
        lo, hi = np.searchsorted(mid, A), np.searchsorted(mid, A + L)
        nb = (i1 - i0) + (hi - lo)
        hard = eos | (4 if i1 > i0 else 0) | (16 if L < SIZE else 0)
        if nb < (L // 40):
            hard |= 1
        mx = max(dw[lo:hi].max() if hi > lo else 0, adw[i0:i1].max() if i1 > i0 else 0)
        if mx > MAX_DWELL:
            hard |= 2
        g_lo, g_hi = (t1 - hi, t1 - lo) if minus else (t0 + lo, t0 + hi)
        mlo, mhi = [], []
        for s, t in ivs:
            s, t = max(s, g_lo), min(t, g_hi)
            if s < t:
                mlo.append((i1 - i0) + (g_hi - t if minus else s - g_lo))
                mhi.append((i1 - i0) + (g_hi - s if minus else t - g_lo))
        out[ci] = (a, L, int(g_lo), int(g_hi), int(nb), mlo, mhi, int(hard), int(i0), int(i1))
    return out


def process(job):
    path, out, seg, masks, patch, pod5_paths, flag_part, head_dir, ref, bam = job
    t = time.time()
    name = os.path.basename(path)[:-len(".parquet")]
    raw = pl.read_parquet(path, columns=["read_id", "seg", "n", "f_first", "f_exit", "dwell", "status", "deficit"])
    if flag_part:
        sq = seg.select("read_id", "seg", "q_start", "q_end", "qlen")
        bad = flag_misplaced(raw.drop("dwell"), sq, seg.select("read_id", "num_samples", "trim_start").unique("read_id")).select("read_id", "seg")
        raw = raw.join(bad, on=["read_id", "seg"], how="anti")
    s6 = (raw.filter(pl.col("status") == "ok").drop("status", "deficit").join(seg, on=["read_id", "seg"])).sort("read_id", "seg")
    head = read_aligned(f"{head_dir}/{name}.parquet") if head_dir else {}
    full = read_aligned(f"{head_dir}_full/{name}.parquet") if head_dir else {}
    by_seg = {}
    for m in masks.filter(pl.col("read_id").is_in(s6["read_id"].unique().implode())).iter_rows():
        by_seg.setdefault((m[0], m[1]), []).append((m[3], m[4]))
    patch_by_contig = {}
    for c, s, e_ in patch.iter_rows():
        patch_by_contig.setdefault(c, []).append((s, e_))
    keys = ("read_id", "parent_id", "pod5", "split_start", "num_samples", "win", "win_len", "contig", "strand", "g_lo", "g_hi", "n_bases",
            "mask_lo", "mask_hi", "w", "split", "seg_off", "seg_n", "f_first", "f_exit", "hard", "ad_i0", "ad_i1", "ad_e")
    cols = {k: [] for k in keys}
    unres = {"read_id": [], "win": [], "win_len": [], "kind": []}
    stats = {"reads": 0, "windows": 0, "unresolved": 0, "eos": 0, "short": 0, "head": 0, "stall": 0, "few_bases": 0, "adapter_missing": 0}
    kept, seg_off = [], 0
    fa = None
    writer = None
    if bam:
        import pysam
        fa = pysam.FastaFile(ref)
        os.makedirs(os.path.join(out, "bam"), exist_ok=True)
        writer = pysam.AlignmentFile(os.path.join(out, "bam", name + ".bam.tmp"), "wb", threads=2,
                                     header={"HD": {"VN": "1.6", "SO": "unsorted"}, "CO": [f"window cleaning v3, adapter {ADAPTER}, {name}"]})
    rows_all = list(s6.iter_rows(named=True))
    i = 0
    while i < len(rows_all):
        j = i
        while j < len(rows_all) and rows_all[j]["read_id"] == rows_all[i]["read_id"]:
            j += 1
        rs = rows_all[i:j]
        i = j
        chunks = chunk_read(rs[0]["num_samples"] - DNA_TRIM_START, SIZE, OVERLAP)
        stats["reads"] += 1
        resolved = {}
        last_hi = max(FRAME * (r["f_exit"] + 1) for r in rs)
        for r in rs:
            dwell = np.asarray(r["dwell"], np.int64)
            e_ref, ea = None, None
            if r.get("want_ad"):
                e_ref, ea = adapter_frames(r["read_id"], r["seg"], r["f_first"], r["n"], head, full)
                if ea is None:
                    stats["adapter_missing"] += 1
                    r["_pend"] = True
            e = e_ref if e_ref is not None else _entries({"n": r["n"], "f_first": r["f_first"], "f_exit": r["f_exit"], "dwell": dwell})
            f_first, f_exit = int(e[0]), int(e[-1] - 1)
            r["want_ad"] = bool(r.get("want_ad")) and ea is not None
            res = read_windows(r, e, ea, by_seg.get((r["read_id"], r["seg"]), ()), patch_by_contig, chunks,
                               FRAME * (r["f_exit"] + 1) >= last_hi)
            if not res:
                continue
            split = "heldout" if r["contig"].startswith(HOLDOUT) else r["split"]
            flat = np.minimum(np.diff(e), 65535).astype(np.uint16)
            new = False
            for ci, row in res.items():
                if ci in resolved:
                    continue
                resolved[ci] = 1
                new = True
                a, L, g_lo, g_hi, nb, mlo, mhi, hard, i0, i1 = row
                for k, v in (("read_id", r["read_id"]), ("parent_id", r["parent_id"]), ("pod5", pod5_paths[r["pod5_file"]]),
                             ("split_start", r["split_start"]), ("num_samples", r["num_samples"]), ("win", a), ("win_len", L),
                             ("contig", r["contig"]), ("strand", r["strand"]), ("g_lo", g_lo), ("g_hi", g_hi), ("n_bases", nb),
                             ("mask_lo", mlo), ("mask_hi", mhi), ("w", r["w_channel"]), ("split", split), ("seg_off", seg_off),
                             ("seg_n", r["n"]), ("f_first", f_first), ("f_exit", f_exit), ("hard", hard), ("ad_i0", i0), ("ad_i1", i1),
                             ("ad_e", [int(x) for x in ea] if i1 > i0 else None)):
                    cols[k].append(v)
                for bit, key in ((8, "eos"), (16, "short"), (4, "head"), (2, "stall"), (1, "few_bases")):
                    stats[key] += bool(hard & bit)
                stats["windows"] += 1
                if writer is not None:
                    import pysam
                    s = fa.fetch(r["contig"], g_lo, g_hi).upper()
                    s = s.translate(_COMP)[::-1] if r["strand"] == "-" else s
                    s = ADAPTER[i0:i1] + s
                    rec = pysam.AlignedSegment(writer.header)
                    rec.query_name, rec.flag, rec.query_sequence = f"{r['read_id']}:{a}", 4, s or "N"
                    rec.set_tags([("ct", r["contig"]), ("ts", int(g_lo)), ("te", int(g_hi)), ("sd", r["strand"]), ("wn", int(a)), ("wl", int(L)),
                                  ("ad", int(i1 - i0)), ("hd", int(hard)), ("w", float(r["w_channel"]))])
                    writer.write(rec)
            if new:
                kept.append(flat)
                seg_off += len(flat)
        # chunks nobody resolved -> unresolved table, with the reason
        for ci, (a, b) in enumerate(chunks):
            if ci in resolved:
                continue
            A, Aend = a + DNA_TRIM_START, b + DNA_TRIM_START
            touch = [r for r in rs if FRAME * r["f_first"] < Aend and FRAME * (r["f_exit"] + 1) > A]
            if ci == 0 and any(r.get("_pend") for r in rs):
                kind = "head_pending"
            elif len(touch) >= 2:
                kind = "junction"
            elif ci == len(chunks) - 1 and touch and Aend - FRAME * (touch[0]["f_exit"] + 1) > END_TOL:
                kind = "tail"
            elif not touch and Aend > last_hi:
                kind = "tail"
            else:
                kind = "unaligned"
            unres["read_id"].append(rs[0]["read_id"]); unres["win"].append(a); unres["win_len"].append(b - a); unres["kind"].append(kind)
            stats["unresolved"] += 1
    df = pl.DataFrame(cols, schema_overrides={"win": pl.Int32, "win_len": pl.Int32, "n_bases": pl.Int32, "w": pl.Float32,
                                              "mask_lo": pl.List(pl.Int32), "mask_hi": pl.List(pl.Int32), "seg_off": pl.Int64,
                                              "seg_n": pl.Int32, "f_first": pl.Int32, "f_exit": pl.Int32, "hard": pl.Int8,
                                              "ad_i0": pl.Int8, "ad_i1": pl.Int8, "ad_e": pl.List(pl.Int32)})
    df = df.with_columns(pl.lit(None, pl.Utf8).alias("label"), pl.lit(None, pl.List(pl.Float32)).alias("lw"),
                         pl.lit(None, pl.List(pl.Float32)).alias("lc"))
    os.makedirs(os.path.join(out, "dwell"), exist_ok=True)
    np.save(os.path.join(out, "dwell", name + ".npy"), np.concatenate(kept or [np.empty(0, np.uint16)]))
    df.write_parquet(os.path.join(out, name + ".parquet"))
    pl.DataFrame(unres, schema_overrides={"win": pl.Int32, "win_len": pl.Int32}).write_parquet(os.path.join(out, name + ".unresolved.parquet"))
    if writer is not None:
        writer.close()
        os.replace(os.path.join(out, "bam", name + ".bam.tmp"), os.path.join(out, "bam", name + ".bam"))
    return {"file": name, "segments": s6.height, **stats, "bases": int(df["n_bases"].sum() or 0), "seconds": round(time.time() - t, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--part", default="part1", choices=sorted(PARTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="only the first N stage-6 files (testing)")
    ap.add_argument("--files", default="", help="comma list of pod5 file names (testing)")
    ap.add_argument("--stage6", default=None, help="stage-6 dir (default: the part's; use stage6_*_long for the second pass)")
    ap.add_argument("--bam", action="store_true", help="also write one window BAM per pod5 file")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t = time.time()
    seg, masks, patch = load_meta(args.part)
    seg = seg.join(want_adapter(args.part), on=["read_id", "seg"], how="left").with_columns(pl.col("want_ad").fill_null(False))
    pod5_paths = {os.path.basename(p): p for p in glob.glob(PARTS[args.part]["pod5"])}
    s6dir = args.stage6 or f"{BD}/{PARTS[args.part]['stage6']}"
    files = sorted(glob.glob(f"{s6dir}/*.parquet"))
    if args.files:
        keep = set(args.files.split(","))
        files = [f for f in files if os.path.basename(f)[:-len('.parquet')] in keep]
    files = files[:args.limit]
    ref = os.path.join(BD, PARTS[args.part]["ref"])
    head_dir = os.path.join(BD, HEAD_DIRS[args.part]) if args.part in HEAD_DIRS else None
    jobs = []
    for f in files:
        pod5 = os.path.basename(f)[:-len(".parquet")]
        if os.path.exists(os.path.join(args.out, pod5 + ".unresolved.parquet")):
            continue                                       # resumable: finished files are skipped
        part = seg.filter(pl.col("pod5_file") == pod5)
        jobs.append((f, args.out, part, masks.filter(pl.col("read_id").is_in(part["read_id"].implode())), patch, pod5_paths,
                     args.part != "part1", head_dir, ref, args.bam))
    print(f"metadata {seg.height} segments, {len(jobs)} files to do, head dir {head_dir}, {time.time() - t:.0f} s", flush=True)
    stats = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for s in pool.imap_unordered(process, jobs):
            stats.append(s)
            print(json.dumps(s), flush=True)
    if stats:
        tot = {k: sum(s[k] for s in stats) for k in stats[0] if k != "file" and k != "seconds"}
        with open(os.path.join(args.out, f"params_{args.part}_{int(time.time())}.json"), "w") as f:
            json.dump({"size": SIZE, "overlap": OVERLAP, "trim": DNA_TRIM_START, "end_tol": END_TOL, "part": args.part, "files": len(stats), **tot}, f, indent=1)
        print(json.dumps(tot), f"total {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()

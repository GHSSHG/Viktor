"""gap windows at the READ EDGES (user chose to continue with gap windows, 2026-10-28).

A gap window has a stretch of signal of more than --tol samples that no aligned base covers. Three situations were mixed in the 'gap' class
(10-file sample, 59,937 windows): read end side 42%, read start side 26%, internal (two loci, or one locus with a hole) 32%.
This pass resolves the two EDGE cases; internal gaps stay unresolved (the sequence of the hole is unknown).

  edge_end    the window contains the end of the read's LAST segment followed by > tol uncovered samples (the read's unaligned tail).
              label = the bases of the segment(s) inside the window, then the stop symbol (hard bit 8).
  edge_start  the window contains the start of the read's FIRST segment preceded by > tol uncovered samples (the unaligned read start).
              label = the bases inside the window; the unaligned lead gets no output (hard bit 64), consistent with the empty_start windows.
  both        the window holds a whole short segment with unaligned stretches on both sides (hard bits 8 + 64).

Conditions: the window touches >= 1 segment, all gaps between touched segments are <= tol samples (and overlaps <= tol), a big lead stretch
belongs to the read's FIRST segment, a big trail stretch to its LAST segment. As for tail_active / start windows, the Dorado calls of the
unaligned stretch (head before the first segment / tail after the last, SUP BAM, used only to LOCATE) are mapped with minimap2 map-ont; a read
whose stretch maps over >= --cov of its length with identity >= --ident (or MAPQ >= 20) looks like real unaligned DNA and its window is EXCLUDED
(<file>.gap_edge_excluded.parquet), not labelled. Stretches shorter than --min_len bases are labelled.

Output per pod5 file in --dir: <file>.gap_edge.parquet (schema of .resolved.parquet, rule edge_end / edge_start / edge_both) and
<file>.gap_edge_excluded.parquet. Reads .still_unresolved.parquet (reason gap); does not modify it.

Usage (from ~/Work/Viktor):  python -m scripts.resolve_gap_edges --part part1 --dir ~/Work/BasecallerData/viktor_win_v3_resolved
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import polars as pl
import pysam

from viktor.chunking import DNA_TRIM_START
from viktor.data import BD, PARTS

sys.path.insert(0, os.path.join(BD, "scripts"))
from placement import flag_misplaced  # noqa: E402

from scripts.make_windows import HOLDOUT, load_meta  # noqa: E402
from scripts.make_windows_v3 import FRAME, MAX_DWELL, SIZE, _COMP, _entries  # noqa: E402
from scripts.resolve_start_empty import heads_of_file  # noqa: E402
from scripts.resolve_tail_active import tails_of_file  # noqa: E402
from scripts.resolve_windows_v3 import piece  # noqa: E402

pysam.set_verbosity(0)
KEYS = ["read_id", "parent_id", "pod5", "split_start", "num_samples", "win", "win_len", "rule", "n_bases", "label", "mask_lo", "mask_hi",
        "hard", "w", "split", "n_pieces", "uncovered", "contigs", "segs"]
SCH = {"win": pl.Int32, "win_len": pl.Int32, "n_bases": pl.Int32, "hard": pl.Int8, "uncovered": pl.Int32, "n_pieces": pl.Int8, "split_start": pl.Int64,
       "num_samples": pl.Int64, "w": pl.Float32, "mask_lo": pl.List(pl.Int32), "mask_hi": pl.List(pl.Int32), "contigs": pl.List(pl.String),
       "segs": pl.List(pl.Int32)}


def classify_file(job):
    """-> (candidate windows, per-read q coordinates, stats) for one pod5 file. Window dict: all row fields + side."""
    path, dir_, name, seg, masks, patch, pod5_paths, flag_part, ref, tol = job
    u = pl.read_parquet(os.path.join(dir_, name + ".still_unresolved.parquet")).filter(pl.col("reason") == "gap")
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
    segs_of = {}
    for r in s6.iter_rows(named=True):
        segs_of.setdefault(r["read_id"], []).append(r)
    fa = pysam.FastaFile(ref)
    cands, coords = [], {}
    st = {"gap_in": u.height, "edge_candidates": 0, "no_pieces": 0, "internal": 0, "overlap": 0}
    for rid, w, L, _, _ in u.iter_rows():
        rs = segs_of.get(rid)
        if not rs:
            st["internal"] += 1
            continue
        ents = []
        for r in rs:
            e = _entries({"n": r["n"], "f_first": r["f_first"], "f_exit": r["f_exit"], "dwell": np.asarray(r["dwell"], np.int64)})
            ivs = list(by_seg.get((r["read_id"], r["seg"]), ())) + [(s, t_) for s, t_ in patch_by_contig.get(r["contig"], [])
                                                                    if s < r["t_end"] and t_ > r["t_start"]]
            ents.append((e, r, ivs))
        order = sorted(range(len(ents)), key=lambda i: ents[i][0][0])
        A, B = w + DNA_TRIM_START, w + DNA_TRIM_START + L
        pcs = []
        for i in order:
            e, r, ivs = ents[i]
            p = piece(e, r, A, B, ivs)
            if p is not None:
                pcs.append((i, p, FRAME * int(e[p[0]]), FRAME * int(e[p[1]])))
        if not pcs:
            st["no_pieces"] += 1
            continue
        ok = True
        for k in range(len(pcs) - 1):
            g = pcs[k + 1][2] - pcs[k][3]
            if g > tol:
                ok = False
            elif g < -tol:
                ok = False
                st["overlap"] += 1
        if not ok:
            st["internal"] += 1
            continue
        lead, trail = pcs[0][2] - A, B - pcs[-1][3]
        first_i, last_i = order[0], order[-1]
        lead_big = lead > tol and pcs[0][0] == first_i and FRAME * int(ents[first_i][0][0]) >= A
        trail_big = trail > tol and pcs[-1][0] == last_i and FRAME * int(ents[last_i][0][-1]) <= B
        if (lead > tol and not lead_big) or (trail > tol and not trail_big) or not (lead_big or trail_big):
            st["internal"] += 1
            continue
        lab, mlo, mhi, off, hard = [], [], [], 0, 0
        for i, (lo, hi, g_lo, g_hi, mk, mx), _, _ in pcs:
            r = ents[i][1]
            s = fa.fetch(r["contig"], g_lo, g_hi).upper()
            lab.append(s.translate(_COMP)[::-1] if r["strand"] == "-" else s)
            mlo += [off + a for a, b in mk]
            mhi += [off + b for a, b in mk]
            off += hi - lo
            hard |= 2 if mx > MAX_DWELL else 0
        hard |= (16 if L < SIZE else 0) | (1 if off < L // 40 else 0) | (8 if trail_big else 0) | (64 if lead_big else 0)
        c0 = ents[pcs[0][0]][1]
        rule = "edge_both" if (lead_big and trail_big) else ("edge_start" if lead_big else "edge_end")
        row = (rid, c0["parent_id"], pod5_paths[c0["pod5_file"]], c0["split_start"], c0["num_samples"], w, L, rule, off, "".join(lab), mlo, mhi, hard,
               c0["w_channel"], "heldout" if any(ents[i][1]["contig"].startswith(HOLDOUT) for i, *_ in pcs) else c0["split"], len(pcs),
               int(lead + trail), [ents[i][1]["contig"] for i, *_ in pcs], [int(ents[i][1]["seg"]) for i, *_ in pcs])
        cands.append((row, lead_big, trail_big))
        co = coords.setdefault(rid, {})
        co["q0"] = int(ents[first_i][1]["q_start"])
        co["q_end"] = int(ents[last_i][1]["q_end"])
        co["qlen"] = int(c0["qlen"])
        st["edge_candidates"] += 1
    return name, cands, coords, st


def map_exclude(fasta, lens, threads, cov, ident, min_len):
    paf = subprocess.run(["nice", "-n", "10", os.path.expanduser("~/tools/minimap2"), "-x", "map-ont", "-t", str(threads), "--secondary=no",
                          f"{BD}/reference/hg002v1.2.map-ont.mmi", fasta], capture_output=True, text=True).stdout
    best = {}
    for l in paf.splitlines():
        f = l.split("\t")
        q, qs, qe, m, bl, mq = f[0], int(f[2]), int(f[3]), int(f[9]), int(f[10]), int(f[11])
        if q not in best or (qe - qs) > best[q][0]:
            best[q] = (qe - qs, m / bl, mq)
    return {q: (cl / lens[q], idn, mq) for q, (cl, idn, mq) in best.items()
            if lens.get(q, 0) >= min_len and cl / lens[q] >= cov and (idn >= ident or mq >= 20)}


def main():
    import multiprocessing as mp
    Pool = mp.get_context("spawn").Pool       # polars + fork deadlocks
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="part1", choices=["part1", "part2", "part3"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--stage6", default=None)
    ap.add_argument("--bam", default="/data/hg002_ref/bam_sup")
    ap.add_argument("--tol", type=int, default=120)
    ap.add_argument("--cov", type=float, default=0.5)
    ap.add_argument("--ident", type=float, default=0.9)
    ap.add_argument("--min_len", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    P = PARTS[a.part]["prefix"]
    seg, masks, patch = load_meta(a.part)
    pod5_paths = {os.path.basename(p): p for p in glob.glob(PARTS[a.part]["pod5"])}
    s6dir = a.stage6 or f"{BD}/{PARTS[a.part]['stage6']}"
    names = [os.path.basename(f)[:-len(".still_unresolved.parquet")] for f in sorted(glob.glob(f"{a.dir}/*.still_unresolved.parquet"))][:a.limit]
    ref = os.path.join(BD, PARTS[a.part]["ref"])
    jobs = []
    for nm in names:
        part = seg.filter(pl.col("pod5_file") == nm)
        jobs.append((f"{s6dir}/{nm}.parquet", a.dir, nm, part, masks.filter(pl.col("read_id").is_in(part["read_id"].implode())), patch, pod5_paths,
                     a.part != "part1", ref, a.tol))
    with Pool(a.workers) as pool:
        res = pool.map(classify_file, jobs, chunksize=1)
    tot = {}
    for _, _, _, st in res:
        for k, v in st.items():
            tot[k] = tot.get(k, 0) + v
    print(f"{len(res)} files: {tot}", flush=True)
    b = pl.read_parquet(f"{BD}/inventory/{P}bam_reads.parquet", columns=["read_id", "flowcell", "grch38_contig", "grch38_start"])
    bm = {r["read_id"]: r for r in b.iter_rows(named=True)}
    # sequences to test: head of reads with a lead stretch, tail of reads with a trail stretch
    hj, tj = [], []
    for nm, cands, coords, _ in res:
        hr, tr = {}, {}
        for row, lead_big, trail_big in cands:
            rid, c = row[0], coords[row[0]]
            if lead_big:
                hr[rid] = (rid, bm[rid]["flowcell"], bm[rid]["grch38_contig"], bm[rid]["grch38_start"], c["q0"], c["qlen"])
            if trail_big:
                tr[rid] = (rid, bm[rid]["flowcell"], bm[rid]["grch38_contig"], bm[rid]["grch38_start"], c["q_end"], c["qlen"])
        hj.append((nm + ".gh", a.dir, a.bam, list(hr.values())))
        tj.append((nm + ".gt", a.dir, a.bam, list(tr.values())))
    with Pool(a.workers) as pool:
        hres = pool.map(heads_of_file, hj, chunksize=1)
        tres = pool.map(tails_of_file, tj, chunksize=1)
    excl = {"head": {}, "tail": {}}
    for kind, rr in (("head", hres), ("tail", tres)):
        lens, fa_all = {}, os.path.join(a.dir, f".gap_{kind}_all.fa")
        with open(fa_all, "w") as o:
            for fa, l in rr:
                lens.update(l)
                with open(fa) as i:
                    o.write(i.read())
                os.remove(fa)
        excl[kind] = map_exclude(fa_all, lens, a.threads, a.cov, a.ident, a.min_len)
        os.remove(fa_all)
        print(f"{kind}s tested {len(lens)}, DNA-like (excluded) {len(excl[kind])}", flush=True)
    n_ok = n_ex = 0
    for nm, cands, coords, _ in res:
        ok, ex = [], []
        for row, lead_big, trail_big in cands:
            rid = row[0]
            hit = (excl["head"].get(rid) if lead_big else None) or (excl["tail"].get(rid) if trail_big else None)
            if hit:
                ex.append((rid, row[5], row[6]) + tuple(hit))
            else:
                ok.append(row)
        pl.DataFrame(ok, schema=KEYS, orient="row", schema_overrides=SCH).write_parquet(f"{a.dir}/{nm}.gap_edge.parquet", compression="zstd")
        pl.DataFrame(ex, schema=["read_id", "win", "win_len", "hit_cov", "hit_ident", "hit_mapq"], orient="row",
                     schema_overrides={"win": pl.Int32, "win_len": pl.Int32}).write_parquet(f"{a.dir}/{nm}.gap_edge_excluded.parquet")
        n_ok += len(ok)
        n_ex += len(ex)
    print(f"windows: gap_edge labelled {n_ok}, excluded {n_ex}", flush=True)


if __name__ == "__main__":
    main()

"""start windows (windows entirely before the read's first aligned segment) -> EMPTY label (user decision 2026-10-28: "打上空标签").

The unaligned read start is mostly not genomic: of 1,365 sampled heads (Dorado calls before the first segment, median 624 bases) only 0.7% map to T2T
over >= 50% of their length (identity median 0.49, MAPQ >= 20: 0), and only 0.1% contain the ONT adapter in the first 120 bases.
So the target for these windows is "output nothing": label "" (the stop symbol immediately).

  * windows with win == 0 (first chunk of the read) are left out by default; with --first they are processed too (user decision 2026-10-28). They hold
    the ONT adapter signal; only 8.9% of those reads would get an adapter label from stage6_head.py (first segment starting <= 100 bases into the read),
    the other 91.1% never get one, so the empty label is consistent for them.
  * as for tail_active: the Dorado head (SUP BAM, original orientation, bases before the first segment; used only to LOCATE) is mapped with minimap2
    map-ont; a read whose head maps over >= --cov of its length with identity >= --ident (or MAPQ >= 20) looks like real unaligned DNA -> its start
    windows are EXCLUDED, not labelled. Heads shorter than --min_head bases cannot be tested and are labelled.

Output per pod5 file in --dir: <file>.start_empty.parquet (schema of .resolved.parquet, rule empty_start, hard bit 32 (+16 if short), label "", n_bases 0)
and <file>.start_excluded.parquet (read_id win win_len hit_cov hit_ident hit_mapq). Reads .still_unresolved.parquet (reason start); does not modify it.

Usage (from ~/Work/Viktor):  python -m scripts.resolve_start_empty --part part1 --dir ~/Work/BasecallerData/viktor_win_v3_resolved
"""

import argparse
import glob
import os
import subprocess
from multiprocessing import Pool

import polars as pl
import pysam

from viktor.data import BD, PARTS

from scripts.make_windows import HOLDOUT, load_meta  # noqa: E402

pysam.set_verbosity(0)
SIZE = 12288
KEYS = ["read_id", "parent_id", "pod5", "split_start", "num_samples", "win", "win_len", "rule", "n_bases", "label", "mask_lo", "mask_hi",
        "hard", "w", "split", "n_pieces", "uncovered", "contigs", "segs"]


def heads_of_file(job):
    name, dir_, bam_dir, reads = job            # reads: (read_id, flowcell, grch38_contig, grch38_start, q0, qlen)
    bams, lens = {}, {}
    out = os.path.join(dir_, ".head_tmp_" + name + ".fa")
    with open(out, "w") as fa:
        for rid, fc, contig, start, q0, qlen in reads:
            if fc not in bams:
                bams[fc] = pysam.AlignmentFile(f"{bam_dir}/{fc}/calls.sorted.bam", "rb")
            seq = None
            for y in bams[fc].fetch(contig, start, start + 1):
                if y.query_name == rid and not y.is_secondary and not y.is_supplementary:
                    seq = y.get_forward_sequence()
                    break
            if seq is None or len(seq) != qlen:
                continue
            h = seq[:q0]
            lens[rid] = len(h)
            if len(h) >= 50:
                fa.write(f">{rid}\n{h}\n")
    return out, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="part1", choices=["part1", "part2", "part3"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--bam", default="/data/hg002_ref/bam_sup")
    ap.add_argument("--cov", type=float, default=0.5)
    ap.add_argument("--ident", type=float, default=0.9)
    ap.add_argument("--min_head", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--first", action="store_true", help="the first chunk of the read (win == 0) instead of the later ones; outputs *.start0_*.parquet")
    a = ap.parse_args()
    suf = "start0" if a.first else "start"
    P = PARTS[a.part]["prefix"]
    files = sorted(glob.glob(f"{a.dir}/*.still_unresolved.parquet"))[:a.limit]
    names = [os.path.basename(f)[:-len(".still_unresolved.parquet")] for f in files]
    seg, _, _ = load_meta(a.part)
    pod5_paths = {os.path.basename(p): p for p in glob.glob(PARTS[a.part]["pod5"])}
    b = pl.read_parquet(f"{BD}/inventory/{P}bam_reads.parquet", columns=["read_id", "flowcell", "grch38_contig", "grch38_start"])
    first = seg.sort("read_id", "q_start").group_by("read_id", maintain_order=True).agg(
        pl.col("q_start").first().alias("q0"), pl.col("qlen").first(), pl.col("contig").first().alias("firstc"), pl.col("parent_id").first(),
        pl.col("pod5_file").first(), pl.col("split_start").first(), pl.col("num_samples").first(), pl.col("w_channel").first(),
        pl.col("split").first()).join(b, on="read_id")
    todo, jobs = {}, []
    for f, nm in zip(files, names):
        u = pl.read_parquet(f).filter((pl.col("reason") == "start") & ((pl.col("win") == 0) if a.first else (pl.col("win") > 0)))
        todo[nm] = u
        r = u.select("read_id").unique().join(first, on="read_id")
        jobs.append((nm, a.dir, a.bam, [(x["read_id"], x["flowcell"], x["grch38_contig"], x["grch38_start"], x["q0"], x["qlen"]) for x in r.iter_rows(named=True)]))
    print(f"{len(jobs)} files, {sum(t.height for t in todo.values())} start windows ({suf}, first={a.first})", flush=True)
    with Pool(a.workers) as pool:
        res = pool.map(heads_of_file, jobs, chunksize=1)
    lens = {}
    for _, l in res:
        lens.update(l)
    fa_all = os.path.join(a.dir, ".head_all.fa")
    with open(fa_all, "w") as o:
        for fa, _ in res:
            with open(fa) as i:
                o.write(i.read())
            os.remove(fa)
    print(f"{len(lens)} read heads extracted, mapping", flush=True)
    paf = subprocess.run(["nice", "-n", "10", os.path.expanduser("~/tools/minimap2"), "-x", "map-ont", "-t", str(a.threads), "--secondary=no",
                          f"{BD}/reference/hg002v1.2.map-ont.mmi", fa_all], capture_output=True, text=True).stdout
    best = {}
    for l in paf.splitlines():
        f = l.split("\t")
        q, qs, qe, m, bl, mq = f[0], int(f[2]), int(f[3]), int(f[9]), int(f[10]), int(f[11])
        if q not in best or (qe - qs) > best[q][0]:
            best[q] = (qe - qs, m / bl, mq)
    os.remove(fa_all)
    exclude = {q: (cl / lens[q], idn, mq) for q, (cl, idn, mq) in best.items()
               if lens.get(q, 0) >= a.min_head and cl / lens[q] >= a.cov and (idn >= a.ident or mq >= 20)}
    print(f"heads mapped {len(best)}, excluded (DNA-like) {len(exclude)} reads of {len(lens)}", flush=True)
    meta = {r["read_id"]: r for r in first.iter_rows(named=True)}
    sch = {"win": pl.Int32, "win_len": pl.Int32, "n_bases": pl.Int32, "hard": pl.Int8, "uncovered": pl.Int32, "n_pieces": pl.Int8, "split_start": pl.Int64,
           "num_samples": pl.Int64, "w": pl.Float32, "mask_lo": pl.List(pl.Int32), "mask_hi": pl.List(pl.Int32), "contigs": pl.List(pl.String),
           "segs": pl.List(pl.Int32)}
    tot_e = tot_x = 0
    for nm in names:
        emp, exc = [], []
        for rid, w, wl, _, _ in todo[nm].iter_rows():
            if rid in exclude:
                exc.append((rid, w, wl) + exclude[rid])
                continue
            m = meta[rid]
            emp.append((rid, m["parent_id"], pod5_paths[m["pod5_file"]], m["split_start"], m["num_samples"], w, wl, "empty_start", 0, "", [], [],
                        32 | (16 if wl < SIZE else 0), m["w_channel"], "heldout" if m["firstc"].startswith(HOLDOUT) else m["split"], 0, int(wl), [], []))
        pl.DataFrame(emp, schema=KEYS, orient="row", schema_overrides=sch).write_parquet(f"{a.dir}/{nm}.{suf}_empty.parquet", compression="zstd")
        pl.DataFrame(exc, schema=["read_id", "win", "win_len", "hit_cov", "hit_ident", "hit_mapq"], orient="row",
                     schema_overrides={"win": pl.Int32, "win_len": pl.Int32}).write_parquet(f"{a.dir}/{nm}.{suf}_excluded.parquet")
        tot_e += len(emp)
        tot_x += len(exc)
    print(f"windows: empty_start {tot_e}, excluded {tot_x}", flush=True)


if __name__ == "__main__":
    main()

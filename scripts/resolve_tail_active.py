"""tail_active windows (CLEANING_RULES.md revision 2026-10-28, item 4; user decision 2026-10-28): windows entirely past the last aligned segment
whose signal is NOT flat.

  * the Dorado calls of the read after its last aligned segment (SUP BAM, original read orientation; used only to LOCATE, never as a label) are
    mapped to T2T with minimap2 map-ont. A read whose tail maps to the genome over >= --cov of its length with identity >= --ident (or MAPQ >= 20)
    looks like real unaligned DNA (a lost segment, a second fragment): ALL its tail_active windows are EXCLUDED (not labelled).
    Sampled 2,114 tails: 0.8% (16) have identity >= 0.9; the other hits are low-identity chance matches (median 0.53, MAPQ 0).
  * every other tail_active window is labelled EOS: label "" (the stop symbol straight away), hard bit 8 (+16 if short), rule eos_active.
    Tails shorter than --min_tail Dorado bases cannot be tested and are labelled EOS too.

Output (per pod5 file, next to the other files in --dir): <file>.tail_eos.parquet (same schema as .resolved.parquet; rule eos_active) and
<file>.tail_excluded.parquet (read_id win win_len hit_cov hit_ident hit_mapq). Reads the .still_unresolved.parquet of resolve_windows_v3; does not modify it.

Usage (from ~/Work/Viktor):  python -m scripts.resolve_tail_active --part part1 --dir ~/Work/BasecallerData/viktor_win_v3_resolved
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import polars as pl
import pysam

from viktor.data import BD, PARTS

from scripts.make_windows import HOLDOUT, load_meta  # noqa: E402

pysam.set_verbosity(0)
SIZE = 12288
KEYS = ["read_id", "parent_id", "pod5", "split_start", "num_samples", "win", "win_len", "rule", "n_bases", "label", "mask_lo", "mask_hi",
        "hard", "w", "split", "n_pieces", "uncovered", "contigs", "segs"]


def tails_of_file(job):
    """-> (fasta path or None, {read_id: tail length}) for the reads of one file that have tail_active windows"""
    name, dir_, bam_dir, reads = job            # reads: list of (read_id, flowcell, grch38_contig, grch38_start, q_end, qlen)
    bams = {}
    out, lens = os.path.join(dir_, ".tail_tmp_" + name + ".fa"), {}
    with open(out, "w") as fa:
        for rid, fc, contig, start, qe, qlen in reads:
            if fc not in bams:
                bams[fc] = pysam.AlignmentFile(f"{bam_dir}/{fc}/calls.sorted.bam", "rb")
            seq = None
            for y in bams[fc].fetch(contig, start, start + 1):
                if y.query_name == rid and not y.is_secondary and not y.is_supplementary:
                    seq = y.get_forward_sequence()
                    break
            if seq is None or len(seq) != qlen:
                continue
            tl = seq[qe:]
            lens[rid] = len(tl)
            if len(tl) >= 50:
                fa.write(f">{rid}\n{tl}\n")
    return out, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="part1", choices=["part1", "part2", "part3"])
    ap.add_argument("--dir", required=True, help="resolve_windows_v3 output dir")
    ap.add_argument("--bam", default="/data/hg002_ref/bam_sup")
    ap.add_argument("--cov", type=float, default=0.5)
    ap.add_argument("--ident", type=float, default=0.9)
    ap.add_argument("--min_tail", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    P = PARTS[a.part]["prefix"]
    files = sorted(glob.glob(f"{a.dir}/*.still_unresolved.parquet"))[:a.limit]
    names = [os.path.basename(f)[:-len(".still_unresolved.parquet")] for f in files]
    seg, _, _ = load_meta(a.part)
    pod5_paths = {os.path.basename(p): p for p in glob.glob(PARTS[a.part]["pod5"])}
    b = pl.read_parquet(f"{BD}/inventory/{P}bam_reads.parquet", columns=["read_id", "flowcell", "grch38_contig", "grch38_start"])
    last = seg.sort("read_id", "q_end").group_by("read_id", maintain_order=True).agg(
        pl.col("q_end").last(), pl.col("qlen").first(), pl.col("contig").last().alias("lastc"), pl.col("parent_id").first(),
        pl.col("pod5_file").first(), pl.col("split_start").first(), pl.col("num_samples").first(), pl.col("w_channel").first(),
        pl.col("split").first()).join(b, on="read_id")
    todo = {}
    jobs = []
    for f, nm in zip(files, names):
        u = pl.read_parquet(f).filter(pl.col("reason") == "tail_active")
        todo[nm] = u
        r = u.select("read_id").unique().join(last, on="read_id")
        jobs.append((nm, a.dir, a.bam, [(x["read_id"], x["flowcell"], x["grch38_contig"], x["grch38_start"], x["q_end"], x["qlen"]) for x in r.iter_rows(named=True)]))
    print(f"{len(jobs)} files, {sum(t.height for t in todo.values())} tail_active windows", flush=True)
    with Pool(a.workers) as pool:
        res = pool.map(tails_of_file, jobs, chunksize=1)
    lens = {}
    for _, l in res:
        lens.update(l)
    fa_all = os.path.join(a.dir, ".tail_all.fa")
    with open(fa_all, "w") as o:
        for fa, _ in res:
            with open(fa) as i:
                o.write(i.read())
            os.remove(fa)
    print(f"{len(lens)} read tails extracted, mapping", flush=True)
    paf = subprocess.run(["nice", "-n", "10", os.path.expanduser("~/tools/minimap2"), "-x", "map-ont", "-t", str(a.threads), "--secondary=no",
                          f"{BD}/reference/hg002v1.2.map-ont.mmi", fa_all], capture_output=True, text=True).stdout
    best = {}
    for l in paf.splitlines():
        f = l.split("\t")
        q, qs, qe, m, bl, mq = f[0], int(f[2]), int(f[3]), int(f[9]), int(f[10]), int(f[11])
        if q not in best or (qe - qs) > best[q][0]:
            best[q] = (qe - qs, m / bl, mq)
    os.remove(fa_all)
    exclude = {}
    for q, (cl, idn, mq) in best.items():
        tl = lens.get(q, 0)
        if tl >= a.min_tail and cl / tl >= a.cov and (idn >= a.ident or mq >= 20):
            exclude[q] = (cl / tl, idn, mq)
    print(f"tails mapped {len(best)}, excluded (DNA-like) {len(exclude)} reads of {len(lens)}", flush=True)
    meta = {r["read_id"]: r for r in last.iter_rows(named=True)}
    tot_eos = tot_ex = 0
    for nm in names:
        u = todo[nm]
        eos, exc = [], []
        for rid, w, wl, _, _ in u.iter_rows():
            if rid in exclude:
                exc.append((rid, w, wl) + exclude[rid])
                continue
            m = meta[rid]
            eos.append((rid, m["parent_id"], pod5_paths[m["pod5_file"]], m["split_start"], m["num_samples"], w, wl, "eos_active", 0, "", [], [],
                        8 | (16 if wl < SIZE else 0), m["w_channel"], "heldout" if m["lastc"].startswith(HOLDOUT) else m["split"], 0, int(wl), [], []))
        sch = {"win": pl.Int32, "win_len": pl.Int32, "n_bases": pl.Int32, "hard": pl.Int8, "uncovered": pl.Int32, "n_pieces": pl.Int8, "split_start": pl.Int64,
               "num_samples": pl.Int64, "w": pl.Float32, "mask_lo": pl.List(pl.Int32), "mask_hi": pl.List(pl.Int32), "contigs": pl.List(pl.String),
               "segs": pl.List(pl.Int32)}
        pl.DataFrame(eos, schema=KEYS, orient="row", schema_overrides=sch).write_parquet(f"{a.dir}/{nm}.tail_eos.parquet", compression="zstd")
        pl.DataFrame(exc, schema=["read_id", "win", "win_len", "hit_cov", "hit_ident", "hit_mapq"], orient="row",
                     schema_overrides={"win": pl.Int32, "win_len": pl.Int32}).write_parquet(f"{a.dir}/{nm}.tail_excluded.parquet")
        tot_eos += len(eos)
        tot_ex += len(exc)
    print(f"windows: eos_active {tot_eos}, excluded {tot_ex}", flush=True)


if __name__ == "__main__":
    main()

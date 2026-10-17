"""Training windows from BasecallerData (HG002 part 1): Dorado-style 12288-sample chunks with reference labels.

Labels come from the stage-6 CRF forced alignment of the T2T HG002 Q100 v1.2 reference to each read
segment: base i of a segment enters at CRF frame e_i (f_first + cumsum(dwell)), i.e. child-read sample 6*e_i.
Each read is chunked as Dorado does at inference (10-sample DNA trim, 12288 samples, overlap 600); a full
window is kept if it lies inside one aligned segment, has no stall (> 200 frames on one base) and at least
12288/40 bases. Its label is the bases whose signal midpoint falls inside the window.

Segments: stage-6 status ok, segments_keep, not misplaced_stage6_align; reads in split.parquet
(train/val/test); windows on chr20 become 'heldout'. Masks (het_ambig, chimera_bp_v2, patch_global_v2)
are stored as window-local base intervals (read orientation); the read weight is read_qc.w_channel.

One parquet per pod5 file, one row per window:
  read_id parent_id pod5 split_start num_samples   signal: child read = parent[split_start : +num_samples]
  win                                             window start in the trimmed child signal (x[win : win+12288])
  contig strand g_lo g_hi n_bases                 label = ref[g_lo:g_hi], reverse-complemented if strand '-'
  mask_lo mask_hi w split

Examples:
  python -m scripts.make_windows build --out ~/Work/BasecallerData/viktor_win12288 --workers 16   # CPU only
  CUDA_VISIBLE_DEVICES=0 python -m scripts.make_windows check --out DIR --model-dir MODEL --n 4000
check basecalls sampled windows with the koi encoder + Dorado CRF head + koi beam search and reports the
global identity of each call against its label (wrong coordinates show up as low identity).
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import time

import numpy as np
import polars as pl

from viktor.chunking import DNA_TRIM_START, chunk_read

BD = os.path.expanduser("~/Work/BasecallerData")
POD5_GLOB = "/data_intel/hg002_giab2025.01/pod5/*/*.pod5"
SIZE, OVERLAP, FRAME = 12288, 600, 6  # samples per window, Dorado overlap, samples per CRF frame
MAX_DWELL, MAX_SPB = 200, 40          # stall limit (frames per base), samples per base limit
HOLDOUT = "chr20_"


def load_meta():
    """Per-segment metadata of usable segments, and all masks in reference coordinates."""
    seg = (pl.read_parquet(f"{BD}/inventory/truth_segments_nocigar.parquet",
                           columns=["read_id", "seg", "contig", "strand", "t_start", "t_end"])
           .join(pl.read_parquet(f"{BD}/masks/segments_keep.parquet").filter(pl.col("keep")).drop("keep"),
                 on=["read_id", "seg"])
           .join(pl.read_parquet(f"{BD}/masks/misplaced_stage6_align.parquet", columns=["read_id", "seg"]),
                 on=["read_id", "seg"], how="anti")
           .join(pl.read_parquet(f"{BD}/inventory/bam_reads.parquet",
                                 columns=["read_id", "parent_id", "split_start", "num_samples", "pod5_file"]), on="read_id")
           .join(pl.read_parquet(f"{BD}/inventory/read_qc.parquet", columns=["read_id", "w_channel"]), on="read_id")
           .join(pl.read_parquet(f"{BD}/masks/split.parquet"), on="read_id"))
    masks = pl.concat([pl.read_parquet(f"{BD}/masks/het_ambig.parquet"),
                       pl.read_parquet(f"{BD}/masks/chimera_bp_v2.parquet")])
    patch = pl.read_parquet(f"{BD}/masks/patch_global_v2.parquet")
    return seg, masks, patch


def window_rows(r, dwell, seg_masks, patch):
    """Windows of one segment. r: segment row (dict); dwell: frames of bases 0..n-2. Returns (rows, skip counts)."""
    n = r["n"]
    e = np.empty(n + 1, np.int64)  # CRF frame where each base enters; e[n] = end of the last base
    e[0] = r["f_first"]
    e[1:n] = r["f_first"] + np.cumsum(dwell, dtype=np.int64)
    e[n] = r["f_exit"] + 1
    dw = np.diff(e)
    mid = 3 * (e[:-1] + e[1:])  # signal midpoint of each base, child-read samples
    seg_lo, seg_hi = FRAME * e[0], FRAME * e[n]
    t0, t1, minus = r["t_start"], r["t_end"], r["strand"] == "-"
    ivs = list(seg_masks) + [(s, t) for s, t in patch.get(r["contig"], []) if s < t1 and t > t0]
    rows, skip = [], {"outside": 0, "stall": 0, "few_bases": 0}
    for a, b in chunk_read(r["num_samples"] - DNA_TRIM_START, SIZE, OVERLAP):
        if b - a != SIZE:
            continue
        A = a + DNA_TRIM_START  # window start in child-read samples
        if A < seg_lo or A + SIZE > seg_hi:
            skip["outside"] += 1
            continue
        lo, hi = np.searchsorted(mid, A), np.searchsorted(mid, A + SIZE)
        if hi - lo < SIZE // MAX_SPB:
            skip["few_bases"] += 1
            continue
        if dw[lo:hi].max() > MAX_DWELL:
            skip["stall"] += 1
            continue
        g_lo, g_hi = (t1 - hi, t1 - lo) if minus else (t0 + lo, t0 + hi)
        mlo, mhi = [], []
        for s, t in ivs:
            s, t = max(s, g_lo), min(t, g_hi)
            if s < t:
                mlo.append(g_hi - t if minus else s - g_lo)
                mhi.append(g_hi - s if minus else t - g_lo)
        rows.append((a, g_lo, g_hi, hi - lo, mlo, mhi))
    return rows, skip


def process(job):
    path, out, seg, masks, patch, pod5_paths = job
    t = time.time()
    name = os.path.basename(path)[:-len(".parquet")]
    s6 = (pl.read_parquet(path, columns=["read_id", "seg", "n", "f_first", "f_exit", "dwell", "status"])
          .filter(pl.col("status") == "ok").drop("status").join(seg, on=["read_id", "seg"]))
    lens = s6["dwell"].list.len().fill_null(0).to_numpy().astype(np.int64)
    flat = s6["dwell"].explode().drop_nulls().to_numpy()
    offs = np.concatenate([[0], np.cumsum(lens)])
    by_seg = {}
    for m in masks.filter(pl.col("read_id").is_in(s6["read_id"].unique().implode())).iter_rows():
        by_seg.setdefault((m[0], m[1]), []).append((m[3], m[4]))
    patch_by_contig = {}
    for c, s, e in patch.iter_rows():
        patch_by_contig.setdefault(c, []).append((s, e))
    cols = {k: [] for k in ("read_id", "parent_id", "pod5", "split_start", "num_samples", "win", "contig", "strand",
                            "g_lo", "g_hi", "n_bases", "mask_lo", "mask_hi", "w", "split")}
    skip = {"outside": 0, "stall": 0, "few_bases": 0}
    for i, r in enumerate(s6.drop("dwell").iter_rows(named=True)):
        rows, sk = window_rows(r, flat[offs[i]:offs[i + 1]], by_seg.get((r["read_id"], r["seg"]), ()), patch_by_contig)
        for k in skip:
            skip[k] += sk[k]
        split = "heldout" if r["contig"].startswith(HOLDOUT) else r["split"]
        for a, g_lo, g_hi, nb, mlo, mhi in rows:
            for k, v in (("read_id", r["read_id"]), ("parent_id", r["parent_id"]), ("pod5", pod5_paths[r["pod5_file"]]),
                         ("split_start", r["split_start"]), ("num_samples", r["num_samples"]), ("win", a),
                         ("contig", r["contig"]), ("strand", r["strand"]), ("g_lo", g_lo), ("g_hi", g_hi),
                         ("n_bases", nb), ("mask_lo", mlo), ("mask_hi", mhi), ("w", r["w_channel"]), ("split", split)):
                cols[k].append(v)
    df = pl.DataFrame(cols, schema_overrides={"win": pl.Int32, "n_bases": pl.Int32, "w": pl.Float32,
                                              "mask_lo": pl.List(pl.Int32), "mask_hi": pl.List(pl.Int32)})
    df.write_parquet(os.path.join(out, name + ".parquet"))
    return {"file": name, "segments": s6.height, "windows": df.height, "bases": int(df["n_bases"].sum() or 0),
            **skip, "seconds": round(time.time() - t, 1)}


def check(args):
    import edlib
    import tomllib
    import torch

    from scripts.basecall_crf import CRFHead, beam_search
    from viktor import load_dorado_encoder
    from viktor.data import index_files, load_windows

    files = index_files(args.out)
    rng = np.random.default_rng(0)
    idx = pl.concat([pl.read_parquet(f) for f in rng.choice(files, min(len(files), 8), replace=False)])
    rows = idx.sample(min(args.n, idx.height), seed=0).to_dicts()
    dev = torch.device("cuda")
    enc = load_dorado_encoder(args.model_dir).to(dev, torch.float16).eval().use_koi()
    head = CRFHead(args.model_dir).to(dev).eval()
    with open(os.path.join(args.model_dir, "config.toml"), "rb") as f:
        q = tomllib.load(f)["qscore"]
    wins = load_windows(rows, enc.mean, enc.stdev)
    ident = []
    for i in range(0, len(wins), 256):
        g = wins[i:i + 256]
        x = torch.from_numpy(np.stack([w[0] for w in g])).to(dev)
        with torch.no_grad():
            calls = beam_search(head(enc(x[:, None])), q["scale"], q["bias"])
        for (_, ref, _), (seq, _, _) in zip(g, calls):
            ident.append(1 - edlib.align(seq, ref, mode="NW", task="distance")["editDistance"] / len(ref))
    a = np.array(ident)
    print(json.dumps({"windows": len(a), "mean": round(float(a.mean()), 4), "median": round(float(np.median(a)), 4),
                      "p10": round(float(np.percentile(a, 10)), 4), "lt0.9": round(float((a < 0.9).mean()), 4),
                      "lt0.7": round(float((a < 0.7).mean()), 4)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="build: only the first N stage-6 files (testing)")
    ap.add_argument("--model-dir", help="check: Dorado SUP model dir")
    ap.add_argument("--n", type=int, default=4000, help="check: windows to basecall")
    args = ap.parse_args()
    if args.cmd == "check":
        return check(args)
    os.makedirs(args.out, exist_ok=True)
    t = time.time()
    seg, masks, patch = load_meta()
    pod5_paths = {os.path.basename(p): p for p in glob.glob(POD5_GLOB)}
    files = sorted(glob.glob(f"{BD}/stage6_align/*.parquet"))[:args.limit]
    jobs = []
    for f in files:
        pod5 = os.path.basename(f)[:-len(".parquet")]
        part = seg.filter(pl.col("pod5_file") == pod5)
        jobs.append((f, args.out, part, masks.filter(pl.col("read_id").is_in(part["read_id"].implode())), patch, pod5_paths))
    print(f"metadata {seg.height} segments, {len(files)} files, {time.time() - t:.0f} s", flush=True)
    stats = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for s in pool.imap_unordered(process, jobs):
            stats.append(s)
            print(json.dumps(s), flush=True)
    tot = {k: sum(s[k] for s in stats) for k in ("segments", "windows", "bases", "outside", "stall", "few_bases")}
    with open(os.path.join(args.out, "params.json"), "w") as f:
        json.dump({"size": SIZE, "overlap": OVERLAP, "trim": DNA_TRIM_START, "max_dwell": MAX_DWELL,
                   "max_spb": MAX_SPB, "holdout": HOLDOUT, "files": len(stats), **tot}, f, indent=1)
    print(json.dumps(tot), f"total {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()

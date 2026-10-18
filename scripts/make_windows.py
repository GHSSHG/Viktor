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

external builds an external test set (e.g. the E. faecium EF22 reads) the same way from a pod5 + minimap2 PAF of
its Dorado calls + reference: primary non-chimeric alignments covering >= 90% of the read, CRF forced alignment
with SUPEncoder's code (as stage 6 / SUPEncoder build_bacteria_test.py), then window_rows; at most --per-read
windows per read, --n windows in one parquet (split 'external', no masks, read weight 1):
  CUDA_VISIBLE_DEVICES=3 python -m scripts.make_windows external --pod5 P --paf PAF --ref FASTA --contigs contig_1 \
      --out ~/Work/BasecallerData/viktor_ef22_n1024.parquet --n 1024
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
POD5_GLOB = "/data/hg002_giab2025.01/pod5/*/*.pod5"
SUPENCODER = os.path.expanduser("~/Work/SUPEncoder")  # forced-alignment code (external)
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


def external(args):
    import sys

    import pod5
    import pysam
    import torch
    sys.path.insert(0, SUPENCODER)
    import forced
    import forced_triton as FT
    from crf import SupCRF
    from data9 import MEAN, STDEV
    W, P, WP = 128, 200, 64  # forced-alignment band parameters of build_bacteria_test.py
    comp = bytes.maketrans(b"ACGTN", b"TGCAN")

    rng = np.random.default_rng(args.seed)
    fa = pysam.FastaFile(args.ref)
    clen = dict(zip(fa.references, fa.lengths))
    contigs = set(args.contigs.split(",")) if args.contigs else set(clen)
    n_prim, al = {}, {}
    for line in open(args.paf):
        f = line.rstrip("\n").split("\t")
        tags = dict(t.split(":", 2)[0::2] for t in f[12:] if t.count(":") >= 2)
        if tags.get("tp") in ("P", "I"):
            n_prim[f[0]] = n_prim.get(f[0], 0) + 1
        if tags.get("tp") != "P" or int(f[11]) < 20 or f[5] not in contigs:
            continue
        qlen, qs, qe, ts, te = int(f[1]), int(f[2]), int(f[3]), int(f[7]), int(f[8])
        if te - ts < 2 * SIZE // 10 or (qe - qs) / qlen < 0.9 or ts < 10 or te > clen[f[5]] - 10:
            continue
        al[f[0]] = dict(contig=f[5], strand=f[4], t_start=ts, t_end=te)
    keep = [r for r in al if n_prim[r] == 1]  # several primary-type alignments = chimeric read
    rng.shuffle(keep)
    keep = keep[:args.reads]
    sig = {}
    with pod5.Reader(args.pod5) as rd:
        for rec in rd.reads(selection=keep, missing_ok=True):
            if rec.num_samples <= args.max_samples:
                sig[str(rec.read_id)] = (rec.signal.astype(np.float32) + rec.calibration.offset) * rec.calibration.scale
    order = [r for r in keep if r in sig]
    print(f"alignments {len(al)}, non-chimeric {sum(n_prim[r] == 1 for r in al)}, reads loaded {len(order)}", flush=True)

    dev = torch.device("cuda")
    model = SupCRF().to(dev).eval().to(torch.bfloat16)
    cols = {k: [] for k in ("read_id", "parent_id", "pod5", "split_start", "num_samples", "win", "contig", "strand",
                            "g_lo", "g_hi", "n_bases", "mask_lo", "mask_hi", "w", "split")}
    stats = {"no_band": 0, "dp_fail": 0, "outside": 0, "stall": 0, "few_bases": 0}
    for b0 in range(0, len(order), args.batch):
        if len(cols["win"]) >= args.n:
            break
        ids = order[b0:b0 + args.batch]
        z = [np.clip((sig[r] - MEAN) / STDEV, -8, 8) for r in ids]
        L = max(len(x) for x in z)
        x = torch.zeros(len(ids), L + -L % 12, dtype=torch.bfloat16)
        for k, v in enumerate(z):
            x[k, :len(v)] = torch.from_numpy(v)
        with torch.no_grad():
            sc_all = model(x.to(dev))
        scs = [sc_all[k, :2 * -(-len(z[k]) // 12)].contiguous() for k in range(len(ids))]
        del sc_all
        dec = forced.free_decode(scs, viterbi=FT.free_viterbi)
        segs = []
        for k, r in enumerate(ids):
            g = al[r]
            if g["strand"] == "+":
                full = fa.fetch(g["contig"], g["t_start"] - 5, g["t_end"]).upper().encode()
            else:
                full = fa.fetch(g["contig"], g["t_start"], g["t_end"] + 5).upper().encode().translate(comp)[::-1]
            c, _, ok = forced.band_centres(full[5:].decode(), dec[k][0], dec[k][1])
            if ok:
                segs.append((k, r, full, c))
            else:
                stats["no_band"] += 1
        if not segs:
            continue
        S, W2, Nm = len(segs), 2 * W, max(len(sg[2]) - 5 for sg in segs)
        offs = np.cumsum([0] + [sc.numel() for sc in scs])
        flat = torch.cat([sc.reshape(-1) for sc in scs])
        f0 = np.array([sg[3][0] - W for sg in segs])
        lol = np.array([sg[3][-1] - W for sg in segs])
        Tseg = lol + W2 - f0
        Tm = int(Tseg.max())
        plo, stt, kkk = np.zeros((S, Tm), np.int64), np.zeros((S, Nm), np.int64), np.zeros((S, Nm), np.int64)
        sp, sl, nn = [], [], []
        for q, (k, r, full, c) in enumerate(segs):
            s_, kk, s_prev = forced.ref_states(full)
            stt[q, :len(s_)], kkk[q, :len(s_)] = s_, kk
            nn.append(len(s_))
            plo[q] = forced.position_band(c, f0[q], Tm, WP)
            sp.append(s_prev)
            sl.append(int(s_[-1]))
        T_ = lambda v: torch.from_numpy(np.asarray(v)).to(dev)
        wp = torch.stack([forced.window(scs[sg[0]], int(f0[q]) - 1 - P, P + W2) for q, sg in enumerate(segs)])
        ws = torch.stack([forced.window(scs[sg[0]], int(lol[q]) + 1, W2 + P) for q, sg in enumerate(segs)])
        pre, suf = FT.free_prefix(wp, T_(sp), P), FT.free_suffix(ws, T_(sl), P)
        del wp, ws
        ent, best, _, _ = FT.forced_dp(flat, T_(offs[[sg[0] for sg in segs]]), T_(stt), T_(kkk), T_(nn), T_(f0),
                                       T_([scs[sg[0]].shape[0] for sg in segs]), T_(plo), T_(Tseg), pre, suf, T_(lol),
                                       W2, WP)
        ent, best = ent.cpu().numpy(), best.cpu().numpy()
        for q, (k, r, full, c) in enumerate(segs):
            e = ent[q, :nn[q]].astype(np.int64)
            if (e < 0).any() or (np.diff(e) < 1).any() or best[q] < -1e29:
                stats["dp_fail"] += 1
                continue
            g, n = al[r], nn[q] - 1  # the last base has no exit frame: keep bases 0..n-1, base n's entry ends them
            assert nn[q] == g["t_end"] - g["t_start"]
            t0, t1 = (g["t_start"], g["t_start"] + n) if g["strand"] == "+" else (g["t_end"] - n, g["t_end"])
            seg = {"n": n, "f_first": int(e[0]), "f_exit": int(e[n]) - 1, "num_samples": len(sig[r]),
                   "t_start": t0, "t_end": t1, "strand": g["strand"], "contig": g["contig"]}
            rows, sk = window_rows(seg, np.diff(e[:n]), (), {})
            for key in sk:
                stats[key] += sk[key]
            for i in sorted(rng.permutation(len(rows))[:args.per_read]):
                a, g_lo, g_hi, nb, _, _ = rows[i]
                for key, v in (("read_id", r), ("parent_id", r), ("pod5", os.path.abspath(args.pod5)),
                               ("split_start", 0), ("num_samples", len(sig[r])), ("win", a), ("contig", g["contig"]),
                               ("strand", g["strand"]), ("g_lo", g_lo), ("g_hi", g_hi), ("n_bases", nb),
                               ("mask_lo", []), ("mask_hi", []), ("w", 1.0), ("split", "external")):
                    cols[key].append(v)
        print(f"reads {b0 + len(ids)}, windows {len(cols['win'])}, {stats}", flush=True)
    df = pl.DataFrame(cols, schema_overrides={"win": pl.Int32, "n_bases": pl.Int32, "w": pl.Float32,
                                              "mask_lo": pl.List(pl.Int32), "mask_hi": pl.List(pl.Int32)}).head(args.n)
    df.write_parquet(args.out)
    print(f"{df.height} windows from {df['read_id'].n_unique()} reads -> {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check", "external"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="build: only the first N stage-6 files (testing)")
    ap.add_argument("--model-dir", help="check: Dorado SUP model dir")
    ap.add_argument("--n", type=int, default=4000, help="check: windows to basecall; external: windows to keep")
    ap.add_argument("--pod5", help="external: pod5 file")
    ap.add_argument("--paf", help="external: minimap2 PAF of the reads' Dorado calls")
    ap.add_argument("--ref", help="external: reference FASTA")
    ap.add_argument("--contigs", default="", help="external: comma list of trusted contigs (default all)")
    ap.add_argument("--reads", type=int, default=1500, help="external: reads sampled before alignment")
    ap.add_argument("--per-read", type=int, default=2, help="external: windows per read")
    ap.add_argument("--max-samples", type=int, default=400_000, help="external: skip longer reads")
    ap.add_argument("--batch", type=int, default=4, help="external: reads per forced-alignment batch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "check":
        return check(args)
    if args.cmd == "external":
        return external(args)
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

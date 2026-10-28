"""Real-data depth experiment on HG002 chr20 (held-out windows): how many reads are needed to call a heterozygous SNV?

Truth: the T2T-HG002 maternal and paternal assemblies (BasecallerData/reference/hetdiff.parquet lists their
differences). A window is aligned to the haplotype it comes from; at a heterozygous SNV the paternal reads carry the
alternate allele (reference = maternal), at every other position all reads carry the reference base.

predict  decode every held-out window overlapping a chr20 region with Dorado and Viktor checkpoints
         -> pickle (rows, labels, predictions)
analyze  align each prediction to its label, collect one observed base per (site, read), then for each depth d draw d
         reads per site: the alternate-count threshold k is the smallest count whose false-call rate at homozygous
         sites is <= alpha; sensitivity = P(alternate count >= k) at heterozygous SNV sites.

  CUDA_VISIBLE_DEVICES=3 python -m scripts.depth_real predict --model-dir MODEL --region 40000000 41000000 \
      --out runs/depth_real/preds.pkl --batch 128 runs/gspo1/snap_0000800.pt
  python -m scripts.depth_real analyze --preds runs/depth_real/preds.pkl --region 40000000 41000000 --out runs/depth_real/curves.json
"""

import argparse
import glob
import json
import os
import pickle

import edlib
import numpy as np
import polars as pl
import pysam

from viktor.data import REF

BD = os.path.expanduser("~/Work/BasecallerData")
HETDIFF = f"{BD}/reference/hetdiff.parquet"
INDEX = f"{BD}/viktor_win12288"
MAT, PAT = "chr20_MATERNAL", "chr20_PATERNAL"
SPLITS = ["heldout"]   # None: windows of every split (train, val, test, heldout)


def set_chrom(chrom, splits):
    global MAT, PAT, SPLITS
    MAT, PAT = f"{chrom}_MATERNAL", f"{chrom}_PATERNAL"
    SPLITS = None if splits == "all" else [splits]
CODE = np.full(256, 4, np.int8)
for _i, _c in enumerate(b"ACGT"):
    CODE[_c] = _i
COMP_CODE = np.array([3, 2, 1, 0, 4], np.int8)


def region_records(lo, hi):
    """Maternal-side hetdiff records in [lo, hi) and the paternal interval they span."""
    h = (pl.scan_parquet(HETDIFF).filter((pl.col("contig") == MAT) & (pl.col("end") > lo) & (pl.col("start") < hi))
         .sort("start").collect())
    p = h.filter((pl.col("other_contig") == PAT) & (pl.col("other_start") > 0))
    return h, int(p["other_start"].min()), int(p["other_end"].max())


def window_rows(lo, hi):
    h, plo, phi = region_records(lo, hi)
    parts = []
    for f in sorted(glob.glob(f"{INDEX}/*.pod5.parquet")):
        d = pl.scan_parquet(f)
        if SPLITS is not None:
            d = d.filter(pl.col("split").is_in(SPLITS))
        parts.append(d.filter(((pl.col("contig") == MAT) & (pl.col("g_hi") > lo) & (pl.col("g_lo") < hi))
                              | ((pl.col("contig") == PAT) & (pl.col("g_hi") > plo) & (pl.col("g_lo") < phi))).collect())
    return pl.concat(parts)


def predict(a):
    import torch
    from scripts.eval_distribution import dorado_preds, viktor_preds
    from viktor import Basecaller, get_tokenizer, load_dorado_encoder
    from viktor.data import load_windows

    dev = torch.device("cuda")
    rows = window_rows(*a.region)
    print("windows", rows.height, rows["contig"].value_counts().to_dicts(), flush=True)
    enc = load_dorado_encoder(a.model_dir).to(dev, torch.float16).eval().use_koi()
    rows = rows.to_dicts()
    wins = load_windows(rows, enc.mean, enc.stdev, REF)
    out = {"rows": rows, "labels": [w[1] for w in wins], "preds": {}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out["preds"]["dorado"] = dorado_preds(a.model_dir, enc, wins, dev, a.batch)
    print("dorado done", flush=True)
    pickle.dump(out, open(a.out, "wb"))
    for p in a.snaps:
        s = torch.load(p, map_location="cpu", weights_only=False)
        model = Basecaller(enc, get_tokenizer("base"))
        model.projector.load_state_dict(s["projector"])
        model.decoder.load_state_dict(s["decoder"])
        model.projector.to(dev)
        model.decoder.to(dev)
        out["preds"][f"viktor_{s['step']}"] = viktor_preds(model, wins, dev, a.batch)
        print(p, "done", flush=True)
        pickle.dump(out, open(a.out, "wb"))
        del model
        torch.cuda.empty_cache()


def make_sites(lo, hi, n_hom, seed=0):
    """Het SNV sites and random homozygous sites, with maternal and paternal coordinates and reference codes."""
    fa = pysam.FastaFile(REF)
    h, plo, phi = region_records(lo, hi)
    snv = h.filter((pl.col("kind") == "snv") & (pl.col("start") >= lo) & (pl.col("start") < hi))
    m_pos, p_pos = snv["start"].to_numpy(), snv["other_start"].to_numpy()
    fm = fa.fetch(MAT, lo, hi).upper()
    fp = fa.fetch(PAT, plo - 50, phi + 50).upper()
    ref_m = np.array([CODE[ord(fm[m - lo])] for m in m_pos], np.int8)
    alt_p = np.array([CODE[ord(fp[p - plo + 50])] for p in p_pos], np.int8)
    ok = (ref_m < 4) & (alt_p < 4) & (ref_m != alt_p)
    het = dict(m=m_pos[ok], p=p_pos[ok], ref=ref_m[ok], alt=alt_p[ok])
    # homozygous sites: >= 25 bp from every record, mapped to the paternal contig through the preceding record
    s, e = h["start"].to_numpy(), h["end"].to_numpy()
    os_, oe = h["other_start"].to_numpy(), h["other_end"].to_numpy()
    rng = np.random.default_rng(seed)
    cand = np.sort(rng.choice(np.arange(lo + 30, hi - 30), n_hom * 2, replace=False))
    i = np.searchsorted(s, cand, side="right") - 1           # last record starting at or before the site
    nxt = np.minimum(i + 1, len(s) - 1)
    near = (i >= 0) & (cand - e[np.maximum(i, 0)] >= 25) & (s[nxt] - cand >= 25)
    cand, i = cand[near], i[near]
    p_map = oe[i] + (cand - e[i])
    ref_c = np.array([CODE[ord(fm[c - lo])] for c in cand], np.int8)
    pat_c = np.array([CODE[ord(fp[q - plo + 50])] if plo - 50 <= q < phi + 50 else 4 for q in p_map], np.int8)
    good = (ref_c < 4) & (ref_c == pat_c)
    hom = dict(m=cand[good][:n_hom], p=p_map[good][:n_hom], ref=ref_c[good][:n_hom])
    return het, hom


def window_obs(pred, label):
    """Observed base code (forward reference orientation not yet applied) per label base; 4 = deleted/other."""
    if not pred:
        return np.full(len(label), 4, np.int8)
    r = edlib.align(pred, label, mode="NW", task="path")
    al = edlib.getNiceAlignment(r, pred, label)
    t = np.frombuffer(al["target_aligned"].encode(), np.uint8)
    q = np.frombuffer(al["query_aligned"].encode(), np.uint8)
    return CODE[q[t != ord("-")]]


def collect(rows, labels, preds, het, hom):
    """-> site id, read id, observed code arrays (one observation per site and read). Sites: het 0..H-1, hom H.."""
    H = len(het["m"])
    pos = {MAT: np.concatenate([het["m"], hom["m"]]), PAT: np.concatenate([het["p"], hom["p"]])}
    sid = {c: np.argsort(pos[c], kind="stable") for c in pos}
    spos = {c: pos[c][sid[c]] for c in pos}
    site, read, obs = [], [], []
    rid = {}
    for r, lab, pr in zip(rows, labels, preds):
        c = r["contig"]
        a, b = np.searchsorted(spos[c], [r["g_lo"], r["g_hi"]])
        if a == b:
            continue
        o = window_obs(pr, lab)
        valid = np.ones(len(lab), bool)
        for mlo, mhi in zip(r["mask_lo"], r["mask_hi"]):
            valid[mlo:mhi] = False
        valid &= np.frombuffer(lab.encode(), np.uint8) != ord("N")
        g = spos[c][a:b]
        if r["strand"] == "+":
            idx = g - r["g_lo"]
            code = o[idx]
        else:
            idx = r["g_hi"] - 1 - g
            code = COMP_CODE[o[idx]]
        keep = valid[idx]
        site.append(sid[c][a:b][keep])
        obs.append(code[keep])
        read.append(np.full(keep.sum(), rid.setdefault(r["read_id"], len(rid)), np.int64))
    site, read, obs = np.concatenate(site), np.concatenate(read), np.concatenate(obs)
    key = site.astype(np.int64) * (len(rid) + 1) + read
    _, first = np.unique(key, return_index=True)
    first.sort()
    return site[first], read[first], obs[first], H


def per_site(site, obs, n_sites):
    order = np.argsort(site, kind="stable")
    site, obs = site[order], obs[order]
    cnt = np.bincount(site, minlength=n_sites)
    start = np.concatenate([[0], np.cumsum(cnt)])
    return [obs[start[i]:start[i + 1]] for i in range(n_sites)], cnt


def draw_counts(cols, ids, d, reps, alt, seed):
    """For each site id (all with coverage >= d): count of reads showing the alternate base in `reps` random draws of
    d reads without replacement."""
    out = np.zeros((len(ids), reps), np.int16)
    n = np.array([len(cols[i]) for i in ids])
    for cov in np.unique(n):
        sel = np.where(n == cov)[0]
        mat = np.stack([cols[ids[j]] for j in sel])
        rng = np.random.default_rng(seed + int(cov) * 1000 + d)
        for rep in range(reps):
            pick = np.argsort(rng.random(mat.shape), axis=1)[:, :d]
            s = np.take_along_axis(mat, pick, axis=1)
            out[sel, rep] = (s == alt[ids[sel]][:, None]).sum(1)
    return out


def draw_false(cols, ids, ref, d, reps, seed):
    """False-call statistic at hom sites: the largest count of any non-reference base in a draw of d reads."""
    out = []
    n = np.array([len(cols[i]) for i in ids])
    for cov in np.unique(n):
        sel = np.where(n == cov)[0]
        mat = np.stack([cols[ids[j]] for j in sel])
        rng = np.random.default_rng(seed + 7 + int(cov) * 1000 + d)
        for _ in range(reps):
            pick = np.argsort(rng.random(mat.shape), axis=1)[:, :d]
            s = np.take_along_axis(mat, pick, axis=1)
            c = np.stack([(s == b).sum(1) for b in range(4)], axis=1)
            c[np.arange(len(sel)), ref[ids[sel]]] = 0
            out.append(c.max(1))
    return np.concatenate(out) if out else np.zeros(1, np.int16)


def analyze(a):
    d = pickle.load(open(a.preds, "rb"))
    het, hom = make_sites(*a.region, a.n_hom)
    H, M = len(het["m"]), len(hom["m"])
    print("het sites", H, "hom sites", M, flush=True)
    depths = [int(x) for x in a.depths.split(",")]
    res = {"region": a.region, "het_sites": H, "hom_sites": M, "alphas": a.alphas, "models": {}}
    for name, preds in d["preds"].items():
        site, read, obs, _ = collect(d["rows"], d["labels"], preds, het, hom)
        cols, cnt = per_site(site, obs, H + M)
        ref_all = np.concatenate([het["ref"], hom["ref"]])
        r = {"coverage_het_median": float(np.median(cnt[:H])), "depth": {}}
        # raw observation accuracy at hom sites and het sites (reference allele)
        ho = np.concatenate(cols[H:]) if M else np.array([])
        r["hom_obs_error"] = float((ho != np.repeat(ref_all[H:], cnt[H:])).mean())
        for dep in depths:
            ids_h = np.where(cnt[:H] >= dep)[0]
            ids_m = H + np.where(cnt[H:] >= dep)[0]
            ch = draw_counts(cols, ids_h, dep, a.reps_het, het["alt"], a.seed)
            fp = draw_false(cols, ids_m, ref_all, dep, a.reps_hom, a.seed)
            e = {"n_het": int(len(ids_h)), "n_hom": int(len(ids_m)), "fp_trials": int(len(fp))}
            for al in a.alphas:
                k = 1
                while (fp >= k).mean() > al:
                    k += 1
                p_site = (ch >= k).mean(1)
                boot = np.random.default_rng(0).choice(p_site, (300, len(p_site))).mean(1) if len(p_site) else np.zeros(1)
                e[str(al)] = {"k": k, "fp": float((fp >= k).mean()), "sens": float(p_site.mean()),
                              "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
            r["depth"][dep] = e
            print(name, dep, json.dumps(e), flush=True)
        res["models"][name] = r
    json.dump(res, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("predict")
    p.add_argument("snaps", nargs="*")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--region", type=int, nargs=2, required=True, help="maternal interval on --chrom")
    p.add_argument("--chrom", default="chr20")
    p.add_argument("--splits", default="heldout", help="index split, or all")
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=128)
    q = sp.add_parser("analyze")
    q.add_argument("--preds", required=True)
    q.add_argument("--region", type=int, nargs=2, required=True)
    q.add_argument("--chrom", default="chr20")
    q.add_argument("--out", required=True)
    q.add_argument("--depths", default="4,6,8,10,12,15,20,25")
    q.add_argument("--alphas", type=float, nargs="+", default=[1e-4, 1e-5])
    q.add_argument("--n-hom", type=int, default=100000)
    q.add_argument("--reps-het", type=int, default=20)
    q.add_argument("--reps-hom", type=int, default=5)
    q.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    set_chrom(a.chrom, getattr(a, "splits", "heldout"))
    predict(a) if a.cmd == "predict" else analyze(a)


if __name__ == "__main__":
    main()

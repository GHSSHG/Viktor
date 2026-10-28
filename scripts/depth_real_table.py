"""Depth table for heterozygous SNVs from the predictions written by scripts.depth_real predict.

For each model and depth d: the false-call tail P(max non-reference count >= k) over the homozygous positions is
computed exactly (hypergeometric over each site's real reads); k = smallest count whose tail is <= the limit. Sensitivity
at heterozygous SNV sites is also exact (hypergeometric). 95% CIs: bootstrap over sites, plus the paired difference
(second model - first model) on the same resampled sites.

  python -m scripts.depth_real_table --preds runs/depth_real_chr8/preds.pkl --chrom chr8 --region 70000000 71000000 \
      --out runs/depth_real_chr8/table.json
"""

import argparse
import json
import pickle

import numpy as np
from scipy.stats import hypergeom

import scripts.depth_real as D

LIMITS = [1e-4, 1e-5, 1e-6]
DEPTHS = [6, 8, 10, 12, 15, 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--region", type=int, nargs=2, required=True)
    ap.add_argument("--n-hom", type=int, default=480000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D.set_chrom(a.chrom, "all")
    d = pickle.load(open(a.preds, "rb"))
    het, hom = D.make_sites(*a.region, a.n_hom)
    H, M = len(het["m"]), len(hom["m"])
    print("het", H, "hom", M, flush=True)
    names = list(d["preds"])
    base, other = names[0], names[1]
    cell = {}
    for name in names:
        site, read, obs, _ = D.collect(d["rows"], d["labels"], d["preds"][name], het, hom)
        cols, cnt = D.per_site(site, obs, H + M)
        ref = np.concatenate([het["ref"], hom["ref"]])
        for dep in DEPTHS:
            ok = np.where(cnt[H:] >= dep)[0] + H
            n = cnt[ok]
            m = np.stack([np.array([(cols[i] == b).sum() for i in ok]) for b in range(4)], axis=1)
            m[np.arange(len(ok)), ref[ok]] = 0
            tail = {k: float(np.minimum(sum(hypergeom.sf(k - 1, n, m[:, b], dep) for b in range(4)), 1).mean()) for k in range(1, 10)}
            nh = cnt[:H]
            ah = np.array([(cols[i] == het["alt"][i]).sum() for i in range(H)])
            for lim in LIMITS:
                k = next(k for k in tail if tail[k] <= lim)
                p = np.where(nh >= dep, hypergeom.sf(k - 1, nh, ah, dep), np.nan)
                cell[(name, dep, lim)] = (k, tail[k], p)
        print(name, "coverage median", float(np.median(cnt[:H])), "hom error", float((np.concatenate(cols[H:]) != np.repeat(ref[H:], cnt[H:])).mean()), flush=True)
    rng = np.random.default_rng(0)
    out = {"chrom": a.chrom, "region": a.region, "het_sites": H, "hom_sites": M, "models": [base, other], "cells": {}}
    pct = lambda x: [round(100 * float(np.percentile(x, 2.5)), 1), round(100 * float(np.percentile(x, 97.5)), 1)]
    for dep in DEPTHS:
        for lim in LIMITS:
            kb, fb, pb = cell[(base, dep, lim)]
            ko, fo, po = cell[(other, dep, lim)]
            ok = ~np.isnan(pb) & ~np.isnan(po)
            pb, po = pb[ok], po[ok]
            idx = rng.integers(0, len(pb), (4000, len(pb)))
            bb, bo = pb[idx].mean(1), po[idx].mean(1)
            e = {"sites": int(ok.sum()), "k": [kb, ko], "fp": [fb, fo],
                 base: round(100 * float(pb.mean()), 4), base + "_ci": pct(bb), base + "_missed_per_1000": round(1000 * float(1 - pb.mean()), 2),
                 other: round(100 * float(po.mean()), 4), other + "_ci": pct(bo), other + "_missed_per_1000": round(1000 * float(1 - po.mean()), 2),
                 "diff": round(100 * float((po - pb).mean()), 2), "diff_ci": [round(x, 2) for x in pct(bo - bb)]}
            out["cells"][f"{dep}|{lim}"] = e
            print(f"{dep:>2}x {lim:g} k={e['k']} n={e['sites']}  {base} {e[base]:5.1f} {e[base + '_ci']}  {other} {e[other]:5.1f} {e[other + '_ci']}  diff {e['diff']:+.2f} {e['diff_ci']}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

"""Heterozygous indels in HG002 chr20 (maternal vs paternal assembly), Dorado vs Viktor on the same held-out windows.

For every indel in the region the two local haplotype sequences A (maternal) and B (paternal) are cut with FLANK bases on
each side. A read (window) that covers the whole local interval is assigned to the allele whose sequence is closer, in edit
distance, to the read's predicted bases over that interval (a tie is 'ambiguous'). Maternal reads truly carry A and
paternal reads B. Read level: how often the allele is right. Site level: at depth d call the indel when >= 3 reads and
>= 20% of the assigned reads support B; sensitivity from draws of d reads of the site, false calls from draws of d
maternal-only reads (a homozygous site with the same sequence context).

  python -m scripts.indel_real --preds runs/depth_real/preds.pkl --region 40000000 41000000 --out runs/depth_real/indel.json
"""

import argparse
import json
import pickle
from multiprocessing import Pool

import edlib
import numpy as np
import polars as pl
import pysam

from scripts.depth_real import HETDIFF, MAT, PAT
from viktor.data import REF

FLANK = 12
MAX_SIZE = 40
COMP = str.maketrans("ACGTN", "TGCAN")
rc = lambda s: s.translate(COMP)[::-1]


def make_sites(lo, hi):
    fa = pysam.FastaFile(REF)
    h = (pl.scan_parquet(HETDIFF).filter((pl.col("contig") == MAT) & (pl.col("start") >= lo) & (pl.col("start") < hi)
                                         & pl.col("kind").is_in(["ins", "del"])).sort("start").collect())
    sites = []
    for r in h.iter_rows(named=True):
        size = max(r["end"] - r["start"], r["other_end"] - r["other_start"])
        if size > MAX_SIZE or r["other_contig"] != PAT or r["other_start"] <= 0:
            continue
        gm = (r["start"] - FLANK, r["end"] + FLANK)
        gp = (r["other_start"] - FLANK, r["other_end"] + FLANK)
        A, B = fa.fetch(MAT, *gm).upper(), fa.fetch(PAT, *gp).upper()
        span = (fa.fetch(MAT, r["start"], r["end"]) if r["kind"] == "ins" else fa.fetch(PAT, r["other_start"], r["other_end"])).upper()
        longer = A if len(A) > len(B) else B
        c = span[0]
        uniform = set(span) == {c}
        i = FLANK
        run = len(span)
        j = i + len(span)
        while uniform and j < len(longer) and longer[j] == c:
            run, j = run + 1, j + 1
        j = i - 1
        while uniform and j >= 0 and longer[j] == c:
            run, j = run + 1, j - 1
        sites.append(dict(kind=r["kind"], size=size, hp=bool(uniform and run >= 4), A=A, B=B, gm=gm, gp=gp))
    return sites


def assign_window(args):
    row, pred, lab, cand = args
    """cand: [(site id, (gs, ge), A, B)] sites fully inside this window -> [(site id, read id, assignment)]."""
    out = []
    if not pred:
        return [(s, row["read_id"], 2) for s, *_ in cand]
    res = edlib.align(pred, lab, mode="NW", task="path")
    al = edlib.getNiceAlignment(res, pred, lab)
    t = np.frombuffer(al["target_aligned"].encode(), np.uint8)
    q = np.frombuffer(al["query_aligned"].encode(), np.uint8)
    pos = np.flatnonzero(t != ord("-"))                      # column of each label base
    for s, (gs, ge), A, B in cand:
        if row["strand"] == "+":
            i0, i1 = gs - row["g_lo"], ge - row["g_lo"]
            a, b = A, B
        else:
            i0, i1 = row["g_hi"] - ge, row["g_hi"] - gs
            a, b = rc(A), rc(B)
        seg = q[pos[i0]:pos[i1 - 1] + 1]
        local = bytes(seg[seg != ord("-")]).decode()
        if not local:
            out.append((s, row["read_id"], 2))
            continue
        da = edlib.align(local, a, mode="NW")["editDistance"]
        db = edlib.align(local, b, mode="NW")["editDistance"]
        out.append((s, row["read_id"], 0 if da < db else 1 if db < da else 2))
    return out


def wilson_free_boot(p, n=300):
    p = np.asarray(p, float)
    if not len(p):
        return [float("nan")] * 2
    b = np.random.default_rng(0).choice(p, (n, len(p))).mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--region", type=int, nargs=2, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depths", default="6,8,10,12,15")
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    d = pickle.load(open(a.preds, "rb"))
    sites = make_sites(*a.region)
    print("sites", len(sites), "homopolymer", sum(s["hp"] for s in sites), "size1", sum(s["size"] == 1 for s in sites), flush=True)
    iv = {MAT: [], PAT: []}
    for i, s in enumerate(sites):
        iv[MAT].append((s["gm"][0], s["gm"][1], i))
        iv[PAT].append((s["gp"][0], s["gp"][1], i))
    tasks_by_window = []
    for w, r in enumerate(d["rows"]):
        cand = [(i, (lo, hi), sites[i]["A"], sites[i]["B"]) for lo, hi, i in iv[r["contig"]]
                if lo >= r["g_lo"] and hi <= r["g_hi"]]
        tasks_by_window.append(cand)
    todo = [w for w, c in enumerate(tasks_by_window) if c]
    print("windows to align", len(todo), flush=True)
    res = {"region": a.region, "sites": len(sites), "models": {}}
    depths = [int(x) for x in a.depths.split(",")]
    for name, preds in d["preds"].items():
        with Pool(10) as pool:
            out = pool.map(assign_window, [(d["rows"][w], preds[w], d["labels"][w], tasks_by_window[w]) for w in todo], chunksize=50)
        per = [dict() for _ in sites]          # site -> {read id: (truth, assignment)}
        for w, lst in zip(todo, out):
            truth = 0 if d["rows"][w]["contig"] == MAT else 1
            for s, rid, asg in lst:
                per[s].setdefault(rid, (truth, asg))
        m = {"classes": {}}
        classes = {"all": lambda s: True, "homopolymer": lambda s: s["hp"], "other": lambda s: not s["hp"],
                   "size1": lambda s: s["size"] == 1}
        for cname, f in classes.items():
            tr = np.array([t for i, s in enumerate(sites) if f(s) for t, _ in per[i].values()])
            asg = np.array([x for i, s in enumerate(sites) if f(s) for _, x in per[i].values()])
            ref_r, alt_r = tr == 0, tr == 1
            m["classes"][cname] = {
                "sites": int(sum(f(s) for s in sites)), "reads": int(len(tr)),
                "read_accuracy": float(((asg == tr) & (asg < 2)).mean()),
                "ref_reads_correct": float((asg[ref_r] == 0).mean()), "ref_reads_called_alt": float((asg[ref_r] == 1).mean()),
                "alt_reads_correct": float((asg[alt_r] == 1).mean()), "alt_reads_called_ref": float((asg[alt_r] == 0).mean()),
                "ambiguous": float((asg == 2).mean())}
        m["depth"] = {}
        for dep in depths:
            sens, fps, ids = [], [], []
            for i, s in enumerate(sites):
                reads = list(per[i].values())
                mat = [x for t, x in reads if t == 0]
                if len(reads) < dep or len(mat) < dep:
                    continue
                rng = np.random.default_rng(i * 131 + dep)
                allr = np.array([x for _, x in reads])
                matr = np.array(mat)
                ps, pf = [], []
                for _ in range(a.reps):
                    x = allr[rng.permutation(len(allr))[:dep]]
                    na, nr = (x == 1).sum(), (x == 0).sum()
                    ps.append(na >= 3 and na / max(na + nr, 1) >= 0.2)
                    y = matr[rng.permutation(len(matr))[:dep]]
                    na, nr = (y == 1).sum(), (y == 0).sum()
                    pf.append(na >= 3 and na / max(na + nr, 1) >= 0.2)
                sens.append(np.mean(ps)); fps.append(np.mean(pf)); ids.append(i)
            hp_mask = np.array([sites[i]["hp"] for i in ids])
            m["depth"][dep] = {"sites": len(ids), "sensitivity": float(np.mean(sens)), "sens_ci": wilson_free_boot(sens),
                               "false_call_rate": float(np.mean(fps)),
                               "sens_homopolymer": float(np.mean(np.array(sens)[hp_mask])) if hp_mask.any() else None,
                               "sens_other": float(np.mean(np.array(sens)[~hp_mask])) if (~hp_mask).any() else None}
            print(name, dep, json.dumps(m["depth"][dep]), flush=True)
        res["models"][name] = m
        print(name, json.dumps(m["classes"]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

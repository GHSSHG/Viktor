"""Read-level accuracy from minimap2 PAF (primary alignments): identity = matches / alignment block length.

Example:
  minimap2 -cx map-ont --secondary=no -t 16 REF.mmi calls.fa > calls.paf
  python scripts/score_paf.py dorado=dorado.paf torch=torch.paf koi=koi.paf
"""

import json
import sys

import numpy as np


def score(path):
    ident, frac, matches, blocks, n_reads = [], [], 0, 0, set()
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        n_reads.add(f[0])
        if "tp:A:P" not in f[12:]:
            continue
        qlen, qs, qe, m, b = int(f[1]), int(f[2]), int(f[3]), int(f[9]), int(f[10])
        ident.append(m / b)
        frac.append((qe - qs) / qlen)
        matches += m
        blocks += b
    x = np.array(ident)
    return {"mapped_reads": len(x), "identity_mean": round(100 * x.mean(), 3),
            "identity_median": round(100 * float(np.median(x)), 3),
            "identity_p10": round(100 * float(np.percentile(x, 10)), 3),
            "identity_pooled": round(100 * matches / blocks, 3),
            "aligned_fraction_mean": round(100 * float(np.mean(frac)), 2)}


if __name__ == "__main__":
    res = {}
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        res[name] = score(path)
        print(name, json.dumps(res[name]))

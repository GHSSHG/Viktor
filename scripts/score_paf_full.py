"""Full-read-length identity from minimap2 PAF (primary alignments, needs the cg tag).

Unlike score_paf.py (matches / alignment block), soft-clipped / unaligned query bases count as errors:
  identity_full = matches / (query length + deleted reference bases)
Reads present in --reads (FASTA/FASTQ ids, optional) but without a primary alignment score 0.

Example:
  python scripts/score_paf_full.py dorado=dorado.paf koi=koi.paf [--reads calls.fa]
"""

import json
import re
import sys

import numpy as np

CG = re.compile(r"(\d+)([MIDN])")


def read_ids(path):
    with open(path) as f:
        return {line[1:].split()[0] for line in f if line[0] == ">"}


def score(path, reads=None):
    full, block_id, clip, ident_by_read = [], [], [], {}
    matches = denom = 0
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if "tp:A:P" not in f[12:]:
            continue
        qlen, qs, qe, m, b = int(f[1]), int(f[2]), int(f[3]), int(f[9]), int(f[10])
        cg = next(x[5:] for x in f[12:] if x.startswith("cg:Z:"))
        dels = sum(int(n) for n, op in CG.findall(cg) if op in "DN")
        d = qlen + dels
        ident_by_read[f[0]] = m / d
        block_id.append(m / b)
        clip.append(1 - (qe - qs) / qlen)
        matches += m
        denom += d
    n_unmapped = 0
    if reads is not None:
        for r in reads - ident_by_read.keys():
            ident_by_read[r] = 0.0
            n_unmapped += 1
    x = np.array(list(ident_by_read.values()))
    return {"reads": len(x), "unmapped": n_unmapped,
            "full_mean": round(100 * x.mean(), 3), "full_median": round(100 * float(np.median(x)), 3),
            "full_p10": round(100 * float(np.percentile(x, 10)), 3), "full_pooled": round(100 * matches / denom, 3),
            "block_mean": round(100 * float(np.mean(block_id)), 3), "clipped_frac_mean": round(100 * float(np.mean(clip)), 3)}


if __name__ == "__main__":
    reads, args = None, sys.argv[1:]
    if "--reads" in args:
        i = args.index("--reads")
        reads = read_ids(args[i + 1])
        del args[i:i + 2]
    for arg in args:
        name, path = arg.split("=", 1)
        print(name, json.dumps(score(path, reads)))

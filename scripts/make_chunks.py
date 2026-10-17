"""Cut full Dorado-size chunks from a pod5 file into a .npy for benchmarking.

Signal is converted to pA and standardised with the Dorado model's mean/stdev, then cut into
chunksize windows with step chunksize - overlap (as Dorado does for long reads).

Example:
  python scripts/make_chunks.py --pod5 READS.pod5 --model-dir MODEL --n 2048 --out results/chunks_2048.npy
"""

import argparse
import os
import sys

import numpy as np
import pod5

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod5", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(os.path.join(args.model_dir, "config.toml"), "rb") as f:
        cfg = tomllib.load(f)
    size, overlap = cfg["basecaller"]["chunksize"], cfg["basecaller"]["overlap"]
    mean, stdev = cfg["standardisation"]["mean"], cfg["standardisation"]["stdev"]

    chunks = []
    with pod5.Reader(args.pod5) as reader:
        for rec in reader.reads():
            x = (rec.signal_pa.astype(np.float32) - mean) / stdev
            for start in range(0, len(x) - size + 1, size - overlap):
                chunks.append(x[start:start + size])
                if len(chunks) == args.n:
                    break
            if len(chunks) == args.n:
                break
    if len(chunks) < args.n:
        sys.exit(f"only {len(chunks)} full chunks in {args.pod5}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, np.stack(chunks).astype(np.float16))
    print(f"saved {args.n} x {size} to {args.out}")


if __name__ == "__main__":
    main()

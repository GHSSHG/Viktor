"""Cut Dorado-style chunks from a pod5 file into an .npz for benchmarking.

Signal is scaled and trimmed as Dorado's ScalerNode does (viktor.chunking.dorado_scale), then chunked as Dorado's
variable-chunk-size path does (full chunks + a shorter, repeat-padded last chunk per read).
--full-only keeps only full-size chunks. The .npz holds:
  signal  fp16, all chunks concatenated      lens  samples per (padded) chunk
  valid   real samples per chunk (before repeat padding); the decoder sees only these frames
  new     new (non-overlap, non-padding) samples per chunk, for throughput accounting

Example:
  python scripts/make_chunks.py --pod5 READS.pod5 --model-dir MODEL --n 2048 --out results/chunks_vcs_2048.npz
"""

import argparse
import os
import sys

import numpy as np
import pod5

from viktor.chunking import chunk_read, repeat_pad, scale_record

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod5", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--full-only", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(os.path.join(args.model_dir, "config.toml"), "rb") as f:
        cfg = tomllib.load(f)
    size, overlap = cfg["basecaller"]["chunksize"], cfg["basecaller"]["overlap"]
    mean, stdev = cfg["standardisation"]["mean"], cfg["standardisation"]["stdev"]
    stride = 1
    for l in cfg["model"]["encoder"]["conv"]["sublayers"]:
        stride *= l.get("stride", 1)
    granularity = stride * 64

    chunks, valid, new = [], [], []
    with pod5.Reader(args.pod5) as reader:
        for rec in reader.reads():
            x = scale_record(rec, mean, stdev)
            for i, (a, b) in enumerate(chunk_read(len(x), size, overlap)):
                if args.full_only and b - a != size:
                    continue
                chunks.append(repeat_pad(x[a:b], granularity))
                valid.append(b - a)
                new.append((b - a) - (overlap if i else 0))
                if len(chunks) == args.n:
                    break
            if len(chunks) == args.n:
                break
    if len(chunks) < args.n:
        sys.exit(f"only {len(chunks)} chunks in {args.pod5}")
    lens = np.array([len(c) for c in chunks], dtype=np.int64)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, signal=np.concatenate(chunks).astype(np.float16), lens=lens, valid=np.array(valid), new=np.array(new))
    print(f"saved {args.n} chunks ({(lens == size).sum()} full, {(lens < size).sum()} shorter) to {args.out}")


if __name__ == "__main__":
    main()

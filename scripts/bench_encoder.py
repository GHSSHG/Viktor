"""Encoder-only throughput: koi (Dorado kernels) vs plain PyTorch, on real chunks.

Example:
  CUDA_VISIBLE_DEVICES=0 python scripts/bench_encoder.py --model-dir MODEL --signals results/chunks_2048.npy
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from viktor import load_dorado_encoder


def timed(fn, batches, reps):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        for b in batches:
            fn(b)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / (reps * len(batches))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--signals", required=True)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda")
    enc = load_dorado_encoder(args.model_dir).to(dev, torch.float16).eval()
    x = torch.from_numpy(np.load(args.signals)).to(dev, torch.float16)
    batches = [x[i:i + args.batch, None] for i in range(0, x.shape[0] - args.batch + 1, args.batch)]
    bases_per_chunk = (enc.chunksize - enc.overlap) / 12.5
    res = {"gpu": torch.cuda.get_device_name(), "batch": args.batch, "chunks": len(batches) * args.batch}

    with torch.no_grad():
        ref = enc(batches[0])
        res["torch_ms_per_chunk"] = timed(enc, batches, args.reps) / args.batch * 1e3
        enc.use_koi()
        out = enc(batches[0])
        for _ in range(2):
            enc(batches[0])
        res["koi_ms_per_chunk"] = timed(enc, batches, args.reps) / args.batch * 1e3
    a, b = ref.float().flatten(1), out.float().flatten(1)
    res["koi_vs_torch_cosine_min"] = float(F.cosine_similarity(a, b, dim=1).min())
    res["koi_vs_torch_max_abs_diff"] = float((a - b).abs().max())
    for k in ("torch", "koi"):
        res[f"{k}_bases_per_s"] = round(bases_per_chunk / res[f"{k}_ms_per_chunk"] * 1e3)
        res[f"{k}_ms_per_chunk"] = round(res[f"{k}_ms_per_chunk"], 4)
    res["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(json.dumps(res, indent=2))
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()

"""Throughput: koi encoder (Dorado VCS path), Triton-fused decoder with CUDA-graph decode steps.

Input is an .npz from make_chunks.py (Dorado-style chunks, full-size only or mixed lengths), cut
into consecutive batches of --batch chunks. Decoder weights are random, so each batch runs a fixed
number of decode steps: the expected output length of its longest chunk (+ EOS). Throughput counts
new bases only (overlap and repeat padding excluded), at 12.5 samples per base.

Example:
  CUDA_VISIBLE_DEVICES=0 python scripts/bench.py --model-dir MODEL --tokenizer 4mer --chunks results/chunks_vcs_2048.npz
"""

import argparse
import json
import math
import time

import numpy as np
import torch

from viktor import Basecaller, DecodeEngine, Pipeline, get_tokenizer, load_dorado_encoder

SAMPLES_PER_BASE = 12.5


def timed(fn, reps=1):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokenizer", choices=["base", "4mer"], required=True)
    ap.add_argument("--chunks", required=True, help=".npz from make_chunks.py")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--prefill-batch", type=int, default=128)
    ap.add_argument("--backend", choices=["triton", "torch"], default="triton")
    ap.add_argument("--json", default=None, help="append the result as one JSON line to this file")
    args = ap.parse_args()

    dev = torch.device("cuda")
    tok = get_tokenizer(args.tokenizer)
    encoder = load_dorado_encoder(args.model_dir).to(dev, torch.float16).eval().use_koi()
    model = Basecaller(encoder, tok)
    model.projector.to(dev, torch.bfloat16)
    model.decoder.to(dev, torch.bfloat16)
    model.eval()

    z = np.load(args.chunks)
    sig, lens, new = torch.from_numpy(z["signal"]).to(dev), z["lens"], z["new"]
    offs = np.concatenate([[0], np.cumsum(lens)])
    n_batches = len(lens) // args.batch
    batches, steps, new_bases = [], [], 0.0
    for i in range(n_batches):
        a, b = i * args.batch, (i + 1) * args.batch
        batches.append((sig[offs[a]:offs[b]], lens[a:b].tolist()))
        steps.append(math.ceil(lens[a:b].max() / SAMPLES_PER_BASE / tok.k) + 1)
        new_bases += new[a:b].sum() / SAMPLES_PER_BASE
    n_chunks = n_batches * args.batch

    torch.cuda.reset_peak_memory_stats()
    engine = DecodeEngine(model, args.batch, max(steps), encoder.chunksize // encoder.stride,
                          backend=args.backend, prefill_batch=args.prefill_batch)
    pipe = Pipeline(model, engine)

    with torch.no_grad():
        frames = [encoder.forward_varlen(s, l) for s, l in batches]
        flens = [[n // encoder.stride for n in l] for _, l in batches]
        pipe.run(batches[:2], steps[:2])  # warm-up
        t_enc = timed(lambda: [encoder.forward_varlen(s, l) for s, l in batches])
        t_pre = timed(lambda: [engine.prefill(f, fl) for f, fl in zip(frames, flens)])
        t_all = timed(lambda: [engine.generate(f, fl, st) for f, fl, st in zip(frames, flens, steps)])
    t_seq = timed(lambda: pipe.run(batches, steps, overlap=False))
    t_pipe = timed(lambda: pipe.run(batches, steps, overlap=True))

    res = {
        "gpu": torch.cuda.get_device_name(), "chunks": args.chunks, "tokenizer": args.tokenizer,
        "backend": args.backend, "batch": args.batch, "n_chunks": n_chunks,
        "full_chunks": int((lens[:n_chunks] == encoder.chunksize).sum()), "steps": steps,
        "ms_per_chunk": {k: round(v / n_chunks * 1e3, 4) for k, v in
                         {"encoder": t_enc, "prefill": t_pre, "decode": t_all - t_pre,
                          "sequential": t_seq, "pipeline": t_pipe}.items()},
        "bases_per_s": {"encoder_only": round(new_bases / t_enc), "sequential": round(new_bases / t_seq),
                        "pipeline": round(new_bases / t_pipe)},
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }
    print(json.dumps(res, indent=2))
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()

"""Paired inference benchmarks on exactly one batch of 256 real chunks.

Each variant uses identical weights/data and measures decode directly (prefill excluded).
Run under an external timeout; one process per GPU. No training or model changes.
"""
import argparse
import gc
import json
import math
import statistics
import subprocess
import time

import numpy as np
import torch

from viktor import Basecaller, DecodeEngine, get_tokenizer, load_dorado_encoder


def measure(fn, prepare=None, reps=3):
    samples = []
    for _ in range(reps):
        if prepare:
            prepare()
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples), samples


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--tokenizer", choices=["4mer", "base"], default="4mer")
    ap.add_argument("--variants", default="flash,split,split2,split4")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    torch.manual_seed(123)
    z = np.load(args.chunks)
    lens = z["lens"][:256].tolist()
    assert len(lens) == 256
    signal = torch.from_numpy(z["signal"][:sum(lens)]).cuda()
    new_bases = float(z["new"][:256].sum()) / 12.5
    encoder = load_dorado_encoder(args.model_dir).to("cuda", torch.float16).eval().use_koi()
    model = Basecaller(encoder, get_tokenizer(args.tokenizer))
    model.projector.to("cuda", torch.bfloat16)
    model.decoder.to("cuda", torch.bfloat16)
    model.eval()
    steps = math.ceil(max(lens) / 12.5 / model.tokenizer.k) + 1
    flens = [n // encoder.stride for n in lens]
    frames = encoder.forward_varlen(signal, lens)
    enc_ms, _ = measure(lambda: encoder.forward_varlen(signal, lens), reps=args.reps)
    for variant in args.variants.split(","):
        eng = DecodeEngine(model, 256, steps, max(flens), attention_backend=variant)
        eng.generate(frames, flens, steps)
        torch.cuda.reset_peak_memory_stats()
        pre, pre_samples = measure(lambda: eng.prefill(frames, flens), reps=args.reps)
        dec, dec_samples = measure(lambda: eng.decode(steps), lambda: eng.prefill(frames, flens), args.reps)
        def all_stages():
            f = encoder.forward_varlen(signal, lens)
            return eng.generate(f, flens, steps).to("cpu", copy=True)
        total, total_samples = measure(all_stages, reps=args.reps)
        print(json.dumps(dict(kind="pipeline", commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            gpu=torch.cuda.get_device_name(), tokenizer=args.tokenizer, chunks=args.chunks, n_chunks=256,
            batch=256, steps=steps, variant=variant, ms_per_chunk=dict(encoder=enc_ms/256, prefill=pre/256,
            decode=dec/256, total=total/256), bases_per_s=new_bases/(total/1000),
            samples_ms=dict(prefill=pre_samples, decode=dec_samples, total=total_samples),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)), flush=True)
        del eng
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

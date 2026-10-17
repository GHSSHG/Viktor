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
    ap.add_argument("--variants", default="flash:full,split:full,auto:last")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    assert args.reps > 0
    torch.manual_seed(123)
    z = np.load(args.chunks)
    lens, valid = z["lens"][:256].tolist(), z["valid"][:256].tolist()
    assert len(lens) == 256
    signal = torch.from_numpy(z["signal"][:sum(lens)]).cuda()
    new_bases = float(z["new"][:256].sum()) / 12.5
    encoder = load_dorado_encoder(args.model_dir).to("cuda", torch.float16).eval().use_koi()
    model = Basecaller(encoder, get_tokenizer(args.tokenizer))
    model.projector.to("cuda", torch.bfloat16)
    model.decoder.to("cuda", torch.bfloat16)
    model.eval()
    steps = math.ceil(max(valid) / 12.5 / model.tokenizer.k) + 1
    frames, flens = encoder.encode_chunks(signal, lens, valid)
    enc_ms, _ = measure(lambda: encoder.encode_chunks(signal, lens, valid), reps=args.reps)
    for variant in args.variants.split(","):
        parts = variant.split(":")
        eng = DecodeEngine(model, 256, steps, max(flens), attention_backend=parts[0],
                           prefill_mode=parts[1] if len(parts) > 1 else "full")
        eng.generate(frames, flens, steps)
        torch.cuda.reset_peak_memory_stats()
        pre, pre_samples = measure(lambda: eng.prefill(frames, flens), reps=args.reps)
        dec, dec_samples = measure(lambda: eng.decode(steps), lambda: eng.prefill(frames, flens), args.reps)
        def all_stages():
            f, _ = encoder.encode_chunks(signal, lens, valid)
            return eng.generate(f, flens, steps).to("cpu", copy=True)
        total, total_samples = measure(all_stages, reps=args.reps)
        print(json.dumps(dict(kind="pipeline", commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            gpu=torch.cuda.get_device_name(), tokenizer=args.tokenizer, chunks=args.chunks, n_chunks=256,
            batch=256, steps=steps, variant=variant, attention_backend=eng.fast.attention_backend,
            prefill_mode=eng.prefill_mode, ms_per_chunk=dict(encoder=enc_ms/256, prefill=pre/256,
            decode=dec/256, total=total/256), bases_per_s=new_bases/(total/1000),
            samples_ms=dict(prefill=pre_samples, decode=dec_samples, total=total_samples),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)), flush=True)
        del eng
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

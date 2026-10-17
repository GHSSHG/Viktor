"""Throughput benchmark: encoder, prefill, decode, sequential and pipelined, for one tokenizer.

Decoder weights are random, so the number of decode steps is fixed (--steps) to the expected output
length instead of waiting for EOS. Throughput uses the new bases per chunk of a long read:
(chunksize - overlap) / (sample_rate / bases_per_s) = 935 for SUP v5.2.0.

Example:
  CUDA_VISIBLE_DEVICES=0 python scripts/bench.py --model-dir MODEL --tokenizer 4mer --batch 512
"""

import argparse
import json
import time

import torch

from viktor import Basecaller, DecodeEngine, Pipeline, get_tokenizer, load_dorado_encoder
from viktor import encoder as enc_mod

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
DEFAULT_STEPS = {"base": 984, "4mer": 247}  # ~983 bases per full chunk, plus EOS


def timed(fn, reps):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Dorado model dir, e.g. dna_r10.4.1_e8.2_400bps_sup@v5.2.0")
    ap.add_argument("--tokenizer", choices=["base", "4mer"], required=True)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--steps", type=int, default=None, help="output tokens per chunk (default: base 984, 4mer 247)")
    ap.add_argument("--batches", type=int, default=6, help="batches per timed run")
    ap.add_argument("--enc-dtype", choices=DTYPES, default="fp16")
    ap.add_argument("--dec-dtype", choices=DTYPES, default="bf16")
    ap.add_argument("--prefill-batch", type=int, default=128)
    ap.add_argument("--no-graph", action="store_true", help="disable CUDA graph for decode steps")
    ap.add_argument("--compile", action="store_true", help="torch.compile the decode step")
    ap.add_argument("--json", default=None, help="append the result as one JSON line to this file")
    args = ap.parse_args()

    dev = torch.device("cuda")
    steps = args.steps or DEFAULT_STEPS[args.tokenizer]
    encoder = load_dorado_encoder(args.model_dir).to(dev, DTYPES[args.enc_dtype]).eval()
    model = Basecaller(encoder, get_tokenizer(args.tokenizer))
    model.projector.to(dev, DTYPES[args.dec_dtype])
    model.decoder.to(dev, DTYPES[args.dec_dtype])
    model.eval()

    chunk = encoder.chunksize
    n_frames = chunk // encoder.stride
    bases_per_chunk = (chunk - encoder.overlap) / 12.5
    sig = torch.randn(args.batch, 1, chunk, device=dev, dtype=DTYPES[args.enc_dtype])
    signals = [sig] * args.batches

    torch.cuda.reset_peak_memory_stats()
    engine = DecodeEngine(model, args.batch, steps, n_frames, cuda_graph=not args.no_graph,
                          compile=args.compile, prefill_batch=args.prefill_batch)
    pipe = Pipeline(model, engine)

    with torch.no_grad():
        frames = encoder(sig)
        engine.generate(frames, steps)  # warm-up
        pipe.run(signals[:2], steps)
        t_enc = timed(lambda: encoder(sig), args.batches)
        t_pre = timed(lambda: engine.prefill(frames), args.batches)
        t_dec = timed(lambda: (engine.prefill(frames), engine.decode(steps)), args.batches) - t_pre
    t_seq = timed(lambda: pipe.run(signals, steps, overlap=False), 1) / args.batches
    t_pipe = timed(lambda: pipe.run(signals, steps, overlap=True), 1) / args.batches

    B = args.batch
    res = {
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "flash_attn": enc_mod.flash_attn_func is not None,
        "tokenizer": args.tokenizer, "batch": B, "steps": steps, "enc_dtype": args.enc_dtype,
        "dec_dtype": args.dec_dtype, "cuda_graph": not args.no_graph, "compile": args.compile,
        "ms_per_chunk": {k: round(v / B * 1e3, 4) for k, v in
                         {"encoder": t_enc, "prefill": t_pre, "decode": t_dec,
                          "sequential": t_seq, "pipeline": t_pipe}.items()},
        "bases_per_s": {"encoder_only": round(B * bases_per_chunk / t_enc),
                        "sequential": round(B * bases_per_chunk / t_seq),
                        "pipeline": round(B * bases_per_chunk / t_pipe)},
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }
    print(json.dumps(res, indent=2))
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()

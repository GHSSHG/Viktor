"""Short, graph-captured decode attention comparison; no model changes.

Run one process per GPU, e.g. CUDA_VISIBLE_DEVICES=0 python scripts/bench_attention.py --length 1271.
All variants see identical BF16 inputs and per-sequence lengths. Append writes back the existing
last KV entry, so it implements exactly the same attention as the other variants.
"""
import argparse
import json
import statistics

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache


def graph_time(fn, repeats=5, iterations=30):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return statistics.median(samples), out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=1271)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(123)
    B, H, HK, D, S = 256, 8, 2, 64, args.length
    q = torch.randn(B, 1, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, HK, D, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    rows = torch.arange(B, device="cuda", dtype=torch.int32)
    for mixed in (False, True):
        lens = torch.randint(1, S + 1, (B,), device="cuda", dtype=torch.int32) if mixed else torch.full_like(rows, S)
        lens[0] = S
        if mixed:
            lens[1:4] = torch.tensor([1, min(64, S), min(65, S)], device="cuda", dtype=torch.int32)
        prev = lens - 1
        kn = k[rows.long(), prev.long()][:, None].contiguous()
        vn = v[rows.long(), prev.long()][:, None].contiguous()
        variants = {
            "flash": lambda: flash_attn_with_kvcache(q, k, v, cache_seqlens=lens),
            "split": lambda: flash_attn_with_kvcache(q, k, v, cache_seqlens=lens, cache_batch_idx=rows, num_splits=1),
            "split2": lambda: flash_attn_with_kvcache(q, k, v, cache_seqlens=lens, num_splits=2),
            "split4": lambda: flash_attn_with_kvcache(q, k, v, cache_seqlens=lens, num_splits=4),
            "append": lambda: flash_attn_with_kvcache(q, k, v, k=kn, v=vn, cache_seqlens=prev, num_splits=1),
        }
        mask = torch.arange(S, device="cuda")[None, :] < lens[:4, None]
        ref = F.scaled_dot_product_attention(q[:4].transpose(1, 2).float(),
            k[:4].transpose(1, 2).repeat_interleave(4, 1).float(),
            v[:4].transpose(1, 2).repeat_interleave(4, 1).float(), attn_mask=mask[:, None, None]).transpose(1, 2)
        baseline = variants["flash"]()
        for name, fn in variants.items():
            out = fn()
            torch.testing.assert_close(out[:4].float(), ref, atol=0.008, rtol=0.02)
            torch.testing.assert_close(out, baseline, atol=0.008, rtol=0.02)
            ms, graph_out = graph_time(fn)
            torch.testing.assert_close(graph_out, out, atol=0, rtol=0)
            record = dict(kind="attention", backend=name, length=S, mixed=mixed, batch=B, ms=ms,
                          max_error=float((out[:4].float() - ref).abs().max()))
            if args.profile and not mixed:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    fn()
                    torch.cuda.synchronize()
                record["kernels"] = sorted({e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

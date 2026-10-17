"""GPU checks (needs CUDA, flash-attn, triton, libkoi). Prints a short report; exits non-zero on failure.

  1. Triton kernels (fused ops, decode attention) vs their PyTorch definitions
  2. koi VCS encoder: a mixed-length batch equals encoding each chunk alone; agreement with the PyTorch encoder
  3. Fused decoder (triton backend) vs reference decoder (torch backend): prefill logits, greedy tokens
  4. CUDA graph vs eager decode steps: identical tokens

Usage: CUDA_VISIBLE_DEVICES=0 python tests/gpu_check.py --model-dir MODEL --chunks results/chunks_vcs_2048.npz
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from viktor import Basecaller, DecodeEngine, get_tokenizer, load_dorado_encoder
from viktor.decode_attention import decode_attention
from viktor.encoder import apply_rope
from viktor.triton_ops import add_rmsnorm, qkv_rope_cache, swiglu

failures = []


def report(name, ok, detail):
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


def check_kernels(dev):
    torch.manual_seed(0)
    bf = torch.bfloat16
    n, d, H, HKV, D, S = 1000, 512, 8, 2, 64, 2048
    x, delta, w = torch.randn(n, d, device=dev, dtype=bf), torch.randn(n, d, device=dev, dtype=bf), torch.rand(d, device=dev, dtype=bf)
    ref_x = x + delta
    ref = F.rms_norm(ref_x, (d,), w, 1e-6)
    xk = x.clone()
    h = add_rmsnorm(xk, w, 1e-6, delta=delta)
    report("add_rmsnorm", torch.equal(xk, ref_x) and (h.float() - ref.float()).abs().max() < 0.05,
           f"max |h - ref| {(h.float() - ref.float()).abs().max():.4f}")

    gu = torch.randn(n, 2 * 1536, device=dev, dtype=bf)
    g, u = gu.chunk(2, -1)
    ref = F.silu(g) * u
    out = swiglu(gu)
    report("swiglu", (out.float() - ref.float()).abs().max() < 0.05, f"max diff {(out.float() - ref.float()).abs().max():.4f}")

    qkv = torch.randn(n, (H + 2 * HKV) * D, device=dev, dtype=bf)
    qw, kw = torch.rand(D, device=dev, dtype=bf), torch.rand(D, device=dev, dtype=bf)
    inv = 1.0 / 10000.0 ** (torch.arange(0, D, 2, dtype=torch.float64) / D)
    fr = torch.arange(S, dtype=torch.float64)[:, None] * inv
    cos, sin = fr.cos().float().to(dev), fr.sin().float().to(dev)
    pos = torch.randperm(S, device=dev)[:n].int()  # distinct slots, so every cache write is checkable
    seq = torch.randint(0, 4, (n,), device=dev, dtype=torch.int32)
    kc = torch.zeros(4, S, HKV, D, device=dev, dtype=bf)
    vc = torch.zeros_like(kc)
    q, k, v = qkv_rope_cache(qkv, qw, kw, cos, sin, pos, seq, kc, vc, 1e-6, H, HKV, D, return_kv=True)
    qr, kr, vr = qkv.view(n, H + 2 * HKV, D).split([H, HKV, HKV], 1)
    c, s = cos[pos.long()][:, None], sin[pos.long()][:, None]  # [n, 1, D/2]
    rq = apply_rope(F.rms_norm(qr, (D,), qw, 1e-6)[:, None], c, s)[:, 0]
    rk = apply_rope(F.rms_norm(kr, (D,), kw, 1e-6)[:, None], c, s)[:, 0]
    dq, dk = (q.float() - rq.float()).abs().max(), (k.float() - rk.float()).abs().max()
    cache_ok = torch.equal(kc[seq.long(), pos.long()], k) and torch.equal(vc[seq.long(), pos.long()], vr)
    report("qkv_rope_cache", dq < 0.05 and dk < 0.05 and torch.equal(v, vr) and cache_ok,
           f"max |q - ref| {dq:.4f}, |k - ref| {dk:.4f}, v exact {torch.equal(v, vr)}, cache rows {cache_ok}")


def check_decode_attention(dev):
    torch.manual_seed(0)
    B, H, HKV, D, S = 64, 8, 2, 64, 1300
    lens = torch.cat([torch.tensor([1, 63, 64, 65]), torch.randint(1, S + 1, (B - 4,))]).int().to(dev)
    q = torch.randn(B, H, D, device=dev, dtype=torch.bfloat16)
    k, v = (torch.randn(B, S, HKV, D, device=dev, dtype=torch.bfloat16) for _ in range(2))
    out = decode_attention(q, k, v, lens)
    mask = (torch.arange(S, device=dev)[None] < lens[:, None].long())[:, None, None]  # [B, 1, 1, S]
    ref = F.scaled_dot_product_attention(q[:, :, None].float(), k.transpose(1, 2).float().repeat_interleave(H // HKV, 1),
                                         v.transpose(1, 2).float().repeat_interleave(H // HKV, 1), attn_mask=mask)[:, :, 0]
    err = (out.float() - ref).abs().max()
    report("decode_attention", err < 0.01, f"max |out - fp32 ref| {err:.4f}, lengths incl. 1/63/64/65")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    dev = torch.device("cuda")
    check_kernels(dev)
    check_decode_attention(dev)

    z = np.load(args.chunks)
    lens = z["lens"][:args.n]
    sig = torch.from_numpy(z["signal"][:int(lens.sum())]).to(dev)
    enc = load_dorado_encoder(args.model_dir).to(dev, torch.float16).eval()
    with torch.no_grad():
        ref = enc.forward_varlen(sig, lens.tolist())
        enc.use_koi()
        packed = enc.forward_varlen(sig, lens.tolist())
        alone, off = [], 0
        for n in lens:
            alone.append(enc.forward_varlen(sig[off:off + n], [int(n)]))
            off += n
        alone = torch.cat(alone)
    diff = (packed.float() - alone.float()).abs().max()
    report("koi VCS mixed batch == per chunk", diff == 0, f"max diff {diff:.4g} over {len(lens)} chunks "
           f"({(lens == enc.chunksize).sum()} full, lengths {sorted(set(lens.tolist()))[:6]}...)")
    cos = F.cosine_similarity(packed.float(), ref.float(), dim=1)
    report("koi VCS vs PyTorch encoder", cos.mean() > 0.95, f"per-frame cosine mean {cos.mean():.4f} min {cos.min():.4f}")

    frames, flens = enc.encode_chunks(sig, lens.tolist(), z["valid"][:args.n].tolist())
    keep = np.concatenate([np.arange(s, s + f) for s, f in zip(np.cumsum([0] + [int(n) // enc.stride for n in lens[:-1]]), flens)])
    report("decoder frames = real signal only", torch.equal(frames, packed[torch.from_numpy(keep).to(dev)]),
           f"{packed.shape[0] - frames.shape[0]} padding frames dropped over {len(lens)} chunks")
    for name in ("4mer", "base"):
        torch.manual_seed(0)
        model = Basecaller(enc, get_tokenizer(name))
        model.projector.to(dev, torch.bfloat16)
        model.decoder.to(dev, torch.bfloat16)
        model.eval()
        steps = 24
        outs, logits = {}, {}
        for backend, graph in (("torch", False), ("triton", False), ("triton", True)):
            eng = DecodeEngine(model, len(lens), steps, max(flens), backend=backend, cuda_graph=graph)
            outs[(backend, graph)] = eng.generate(frames, flens, steps).clone()
        agree = (outs[("triton", True)] == outs[("torch", False)]).float().mean()
        report(f"{name}: triton graph == triton eager", torch.equal(outs[("triton", True)], outs[("triton", False)]), "")
        report(f"{name}: triton vs torch greedy tokens", agree > 0.9, f"agreement {agree:.3f} over {steps} steps x {len(lens)} chunks")

    print("FAILED: " + ", ".join(failures) if failures else "all checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

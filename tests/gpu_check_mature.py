"""GPU check of the 4x compression (viktor/mature_encoder.py): exits non-zero on failure.

  1. sliding FlashAttention vs masked SDPA (outputs, input gradients), padding frames zero and without influence
  2. MatureEncoder.encode_chunks on a mixed-length batch equals encoding each chunk alone

Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/gpu_check_mature.py [--model-dir MODEL]
"""

import argparse
import copy

import torch

import viktor.mature_encoder as me
from viktor import load_dorado_encoder

MODEL = "/home/superbig/tools/dorado_models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0"


def check_attention(dev):
    for width, window in [(1024, 64), (2048, 32)]:
        flash = me.BottleneckTransformer(width, window).to(dev)
        dense = copy.deepcopy(flash)
        dense.require_flash = False
        lens = torch.tensor([129, 73, 0], device=dev)
        x = torch.randn(3, 129, width, device=dev, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        fa = me.flash_attn_varlen_func
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = flash(x, lens)
            me.flash_attn_varlen_func = None
            ref = dense(y, lens)
            me.flash_attn_varlen_func = fa
        torch.testing.assert_close(out, ref, atol=0.06, rtol=0.04)
        out.square().mean().backward()
        ref.square().mean().backward()
        torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=0.1)
        assert out[1, 73:].abs().max() == 0 and out[2].abs().max() == 0
        noisy = x.detach().clone()
        noisy[1, 73:] = 1000
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.testing.assert_close(flash(noisy, lens)[1, :73], out[1, :73], atol=1e-5, rtol=1e-5)
        print(f"[ok] compression attention width {width}: flash = SDPA, padding isolated")


def check_batching(dev, model_dir):
    enc = me.MatureEncoder(load_dorado_encoder(model_dir), 4).to(dev, torch.float16).eval()
    g = enc.granularity
    lens = [12288, 12288, 768, 1536 + g]
    sig = torch.randn(sum(lens), device=dev, dtype=torch.float16)
    with torch.no_grad():
        batch, n = enc.encode_chunks(sig, lens, lens)
        off, single = 0, []
        for k in lens:
            f, _ = enc.encode_chunks(sig[off:off + k], [k], [k])
            single.append(f)
            off += k
    single = torch.cat(single)
    cos = torch.nn.functional.cosine_similarity(batch.float(), single.float(), -1)
    assert batch.shape == single.shape and cos.min() > 0.99, (batch.shape, single.shape, cos.min().item())
    print(f"[ok] encode_chunks: mixed batch = one chunk at a time (frames {n}, cosine min {cos.min().item():.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    check_attention(dev)
    check_batching(dev, args.model_dir)

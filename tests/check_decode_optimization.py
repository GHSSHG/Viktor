"""GPU numerical checks for optimized inference, with 256 sequences and eight unchanged layers.

PYTHONPATH=. python tests/check_decode_optimization.py --tokenizer 4mer
"""
import argparse
import json

import torch
from torch import nn

from viktor import Basecaller, DecodeEngine, get_tokenizer


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", choices=["4mer", "base"], default="4mer")
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()
    torch.manual_seed(42)
    enc = nn.Identity()
    enc.d_model = 512
    model = Basecaller(enc, get_tokenizer(args.tokenizer)).cuda().bfloat16().eval()
    lens = [64 * (i % 16 + 1) for i in range(256)]
    frames = torch.randn(sum(lens), 512, device="cuda", dtype=torch.bfloat16)
    steps = 247 if args.tokenizer == "4mer" else 985
    ref = DecodeEngine(model, 256, steps, 1024, cuda_graph=False, attention_backend="flash", prefill_mode="full")
    opt = DecodeEngine(model, 256, steps, 1024, cuda_graph=False, attention_backend=args.backend, prefill_mode="last")
    prefill_logits = []
    handle = model.decoder.lm_head.register_forward_hook(lambda module, inputs, out: prefill_logits.append(out.float().clone()))
    ref.prefill(frames, lens)
    opt.prefill(frames, lens)
    handle.remove()
    torch.testing.assert_close(prefill_logits[1], prefill_logits[0], atol=0.025, rtol=0.025)
    prefill_error = float((prefill_logits[1] - prefill_logits[0]).abs().max())
    # All prefix K/V must be identical: pruning only affects their unused final outputs.
    torch.testing.assert_close(ref.cache.k, opt.cache.k, atol=0, rtol=0)
    torch.testing.assert_close(ref.cache.v, opt.cache.v, atol=0, rtol=0)
    prefill_agree = float((ref.tok == opt.tok).float().mean())
    # Force identical nonterminal inputs. Random models can end immediately, so
    # comparing PAD-heavy greedy outputs alone is not a meaningful numerical check.
    worst = 0.0
    for _ in range(12):
        tokens = torch.randint(3, 3 + model.tokenizer.n_full, (256,), device="cuda")
        logits = []
        for eng in (ref, opt):
            eng.pos32.copy_(eng.pos)
            eng.seqlens.copy_(eng.pos + 1)
            h = eng.fast.decode(model.decoder.embed(tokens), eng.pos32, eng.rows, eng.seqlens, eng.cache)
            logits.append(model.decoder.head(h).float())
            eng.pos.add_(1)
        torch.testing.assert_close(logits[1], logits[0], atol=0.025, rtol=0.025)
        worst = max(worst, float((logits[1] - logits[0]).abs().max()))
    # Full requested generation length, then repeat with a captured step.
    eager = opt.generate(frames, lens, steps).clone()
    opt._capture()
    graph = opt.generate(frames, lens, steps).clone()
    torch.testing.assert_close(eager, graph, atol=0, rtol=0)
    assert torch.isfinite(opt.cache.k).all() and torch.isfinite(opt.cache.v).all()
    # A changed length layout must replace the cached metadata, including a nonzero row offset.
    rev_lens = lens[::-1]
    opt.prefill(frames, rev_lens)
    ref.prefill(frames, rev_lens)
    for b, length in enumerate(rev_lens):
        torch.testing.assert_close(ref.cache.k[:, b, :length+1], opt.cache.k[:, b, :length+1], atol=0, rtol=0)
        torch.testing.assert_close(ref.cache.v[:, b, :length+1], opt.cache.v[:, b, :length+1], atol=0, rtol=0)
    print(json.dumps(dict(status="passed", backend=args.backend, tokenizer=args.tokenizer, batch=256,
                          steps=steps, prefill_token_agreement=prefill_agree, teacher_forced_max_logit_error=worst,
                          prefill_max_logit_error=prefill_error, cache_exact=True, graph_eager_exact=True)), flush=True)


if __name__ == "__main__":
    main()

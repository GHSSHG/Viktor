"""Regression check of the real FAST freeze/compile/unfreeze path (no checkpoint writes).

PYTHONPATH=. torchrun --nproc-per-node 4 tests/gpu_check_unfreeze.py SNAPSHOT
Requires the training dependencies, CUDA and a FAST snapshot.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

from scripts.train import FastTrainable, named_groups, unfreeze_decoder
from viktor import Basecaller
from viktor.fast_encoder import FastEncoder, FastEncoderConfig, ctc_targets
from viktor.model import AlignConfig


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    encoder = FastEncoder(FastEncoderConfig(**state["encoder_cfg"])).to(dev)
    model = Basecaller.from_args(encoder, state["args"]).to(dev)
    for name in ("encoder", "projector", "decoder"):
        getattr(model, name).load_state_dict(state[name])
    model.align = AlignConfig(sigma=1.0)
    trainable = FastTrainable(model)
    opt = torch.optim.AdamW(named_groups(model.decoder, "decoder", 0.1), lr=1e-4, fused=True)
    seqs = ["ACGTTGCA" * 120] * 2
    weights = [np.ones(len(s), np.float32) for s in seqs]
    centers = [np.linspace(0, 255, len(s), dtype=np.float32) for s in seqs]
    ti, ta, w, cs = [t.to(dev) for t in model.batch(seqs, weights, 128, centers)]
    ct, cn = [t.to(dev) for t in ctc_targets(seqs, 512)]
    # Different data on every rank verifies gradient synchronization too.
    torch.manual_seed(29 + rank)
    sig = torch.randn(2, 1, 12288, device=dev)
    lens = torch.full((2,), 12288, device=dev)

    def wrap():
        return torch.compile(torch.nn.parallel.DistributedDataParallel(trainable, device_ids=[dev.index]),
                             dynamic=False)

    def backward(fn):
        trainable.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = fn(sig, lens, ti, ta, w, cs, ct, cn, 0.3, 0.5)[0]
        loss.backward()
        return loss

    model.decoder.requires_grad_(False)
    fn = wrap()
    backward(fn)
    assert all(p.grad is None for p in model.decoder.parameters())
    fn = unfreeze_decoder(model, fn, wrap, compiled=True)
    loss = backward(fn)
    grads = [p.grad for p in model.decoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() and torch.count_nonzero(g) > 0 for g in grads)
    grad_norm = sum(float(g.float().square().sum()) for g in grads) ** 0.5
    # Embeddings have zero decay: a change here proves a gradient update, not decay.
    before = model.decoder.embed.weight.detach().clone()
    opt.step()
    change = (model.decoder.embed.weight.detach() - before).abs().max()
    assert change > 0
    assert all(torch.count_nonzero(s["exp_avg"]) > 0 and torch.count_nonzero(s["exp_avg_sq"]) > 0
               for s in opt.state.values())
    reference = model.decoder.embed.weight.detach().clone()
    dist.broadcast(reference, 0)
    assert torch.equal(reference, model.decoder.embed.weight)
    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, {"rank": rank, "loss": float(loss.detach()), "decoder_grad_norm": grad_norm,
                                  "embedding_max_update": float(change), "adam_states": len(opt.state)})
    if rank == 0:
        print("PASS: frozen decoder stays frozen; unfreezing restores gradients, Adam updates and rank agreement",
              flush=True)
        print(json.dumps(rows, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

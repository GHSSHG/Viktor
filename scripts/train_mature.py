"""Two-GPU mature-encoder study: freeze encoder, train new decoder/interface, then joint fine-tune.

One step is a global weighted-token update across all accumulation microbatches
and ranks. CE and alignment use their own global denominators. Encoder runs eager
with FlashAttention/checkpointing; only decoder loss is compiled. Its parameters
remain trainable throughout. DDP and compiled loss are rebuilt at unfreezing.
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from viktor import Basecaller, load_dorado_encoder
from viktor.mature_encoder import MatureEncoder
from viktor.model import AlignConfig
from viktor.data import WindowDataset, MixedDataset, collate, part_paths
from scripts.train import (parse_source, fixed_windows, named_groups, wsd, temperature_guard,
                           heldout_calls, sharded_calls, identity_stats)


def convert_centers(centers, compression):
    return [np.where(c >= 0, np.maximum((c + .5) / compression - .5, 0), -1).astype(np.float32) for c in centers]


class Trainable(nn.Module):
    def __init__(self, model, compiled=True):
        super().__init__()
        self.model = model
        self.loss_fn = [torch.compile(model.weighted_loss, dynamic=False) if compiled else model.weighted_loss]

    def forward(self, sig, valid, ti, targets, weights, centers):
        frames, lens = self.model.encode(sig, valid)
        return self.loss_fn[0](frames, ti, targets, weights, centers, lens)


def new_groups(model, decay):
    return (named_groups(model.encoder.stages, 'compression', decay)
            + named_groups(model.projector, 'projector', decay)
            + named_groups(model.decoder, 'decoder', decay))


def build_model(args, dev):
    base = load_dorado_encoder(args.model_dir, load_weights=False)
    state = torch.load(args.encoder_init, map_location='cpu', weights_only=False)
    base.load_state_dict(state['encoder'], strict=True)
    del state
    encoder = MatureEncoder(base, args.compression)
    model = Basecaller.from_args(encoder, vars(args))
    # Architecture-independent module seeds make the new decoder exactly equal.
    for root, net in [('projector', model.projector), ('decoder', model.decoder), ('compression', encoder.stages)]:
        for name, layer in net.named_modules():
            if isinstance(layer, (nn.Linear, nn.Embedding, nn.Conv1d, nn.RMSNorm)):
                seed = int.from_bytes(hashlib.sha256(f'{args.seed}:{root}.{name}'.encode()).digest()[:8], 'little') % (2**63-1)
                with torch.random.fork_rng(devices=[]):
                    torch.default_generator.manual_seed(seed)
                    layer.reset_parameters()
    model.align = AlignConfig((1, 3), (0, 4), args.align_sigma, .1)
    return model.to(dev)


def verify_gradients(model, opt, joint):
    watched = {'decoder': model.decoder.embed.weight, 'projector': model.projector[0].weight}
    if model.encoder.compression == 4:
        watched['compression'] = model.encoder.stages[0].transformer.qnorm.weight
    if joint:
        watched['encoder'] = model.encoder.base.conv[0].conv.weight
    for name, module in [('decoder', model.decoder), ('projector', model.projector),
                         ('compression', model.encoder.stages), ('encoder', model.encoder.base)]:
        if name == 'encoder' and not joint:
            assert all(p.grad is None for p in module.parameters())
            continue
        for n, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().max() == 0:
                raise RuntimeError(f'{name}.{n}: missing, nonfinite or zero gradient')
    for name, p in watched.items():
        if name in ('decoder', 'compression'):
            group = next(g for g in opt.param_groups if any(q is p for q in g['params']))
            assert group['weight_decay'] == 0, name
    return {name: (p, p.detach().clone()) for name, p in watched.items()}


def verify_updates(watched, opt):
    for name, (p, before) in watched.items():
        state = opt.state[p]
        change = (p.detach() - before).abs().max()
        assert torch.isfinite(change) and change > 0, f'{name}: no actual update'
        group = next(g for g in opt.param_groups if any(q is p for q in g['params']))
        decay_only = before * (1 - group['lr'] * group['weight_decay'])
        assert (p.detach() - decay_only).abs().max() > 0, f'{name}: only weight decay changed parameters'
        assert state['exp_avg'].abs().max() > 0 and state['exp_avg_sq'].max() > 0
        # A shared value must match across ranks, not just have a nonzero gradient.
        sample = p.detach().flatten()[:256].clone()
        reference = sample.clone()
        dist.broadcast(reference, 0)
        assert torch.allclose(sample, reference, atol=1e-7, rtol=1e-6), f'{name}: ranks diverged'
    if dist.get_rank() == 0:
        print('VERIFIED gradients, real Adam updates, and DDP synchronization: ' + ', '.join(watched), flush=True)


@torch.no_grad()
def evaluate(model, val, batch, dev):
    model.eval()
    stats = {}
    for key, windows in val.items():
        sums = torch.zeros(6, device=dev, dtype=torch.float64)
        mine = windows[dist.get_rank()::dist.get_world_size()]
        for i in range(0, len(mine), batch):
            sig, valid, seqs, ws = collate(mine[i:i+batch])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                frames, lens = model.encode(sig.to(dev), valid)
                _, ce, _, correct, scored = model.weighted_loss(frames, *(x.to(dev) for x in model.batch(seqs, ws)), plens=lens)
            denom = sum(float(np.min(np.pad(w, (0, (-len(w)) % 4), constant_values=np.inf).reshape(-1, 4), axis=1).sum()) + 1 for w in ws)
            sums[:3] += torch.stack([ce.double() * denom, correct.double(), scored.double()])
            sums[3] += denom
        dist.all_reduce(sums)
        stats[f'val/{key}/ce'] = float(sums[0] / sums[3])
        stats[f'val/{key}/acc'] = float(sums[1] / sums[2])
    model.train()
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--encoder-init', required=True)
    p.add_argument('--compression', type=int, choices=[1, 4], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--windows', nargs='+', required=True)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--accum', type=int, default=2)
    p.add_argument('--max-steps', type=int, default=200000)
    p.add_argument('--freeze-encoder-steps', type=int, default=20000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--encoder-lr', type=float, default=1e-5)
    p.add_argument('--warmup', type=int, default=2000)
    p.add_argument('--encoder-warmup', type=int, default=2000)
    p.add_argument('--decay-start', type=int, default=160000)
    p.add_argument('--decay-steps', type=int, default=40000)
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--clip', type=float, default=1.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--pad-to', type=int, default=128)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--val-every', type=int, default=10000)
    p.add_argument('--ckpt-every', type=int, default=20000)
    p.add_argument('--eval-n', type=int, default=256)
    p.add_argument('--heldout-n', type=int, default=512)
    p.add_argument('--eval-batch', type=int, default=8)
    p.add_argument('--no-heldout', action='store_true')
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--wandb-mode', choices=['online', 'disabled'], default='online')
    args = p.parse_args()
    args.encoder_arch, args.tokenizer, args.layers = 'mature', '4mer', 4
    args.prefix_pos_scale, args.align_sigma = args.compression, float(4 / args.compression)
    total = args.decay_start + args.decay_steps
    stop = min(args.max_steps, total)
    dist.init_process_group('nccl', timeout=datetime.timedelta(hours=1))
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(dev)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 64
    # This server's Triton compiler fails on Inductor split-reduction kernels;
    # use the established non-split reduction path, retaining compiled matmuls.
    torch._inductor.config.split_reductions = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(args, dev)
    opt = torch.optim.AdamW(new_groups(model, args.weight_decay), lr=args.lr, betas=(.9, .95), fused=True)
    ckpt = out / 'last.pt'
    step, run_id = 0, None
    if ckpt.exists():
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        for key in ('compression', 'layers', 'tokenizer', 'batch', 'accum', 'freeze_encoder_steps', 'decay_start', 'decay_steps'):
            assert state['args'][key] == vars(args)[key], f'resume differs: {key}'
        for name in ('encoder', 'projector', 'decoder'):
            getattr(model, name).load_state_dict(state[name])
        step, run_id = state['step'], state['wandb_id']
        if state['joint']:
            model.encoder.set_base_trainable(True)
            opt.add_param_group(named_groups(model.encoder.base, 'encoder', args.weight_decay)[0])
            opt.add_param_group(named_groups(model.encoder.base, 'encoder', args.weight_decay)[1])
        opt.load_state_dict(state['opt'])
        del state
    joint = model.encoder.base_trainable

    def wrap():
        return DDP(Trainable(model, not args.no_compile), device_ids=[dev.index], broadcast_buffers=False)

    ddp = wrap()
    sources = [parse_source(s) for s in args.windows]
    datasets = [WindowDataset(d, 'train', model.encoder.mean, model.encoder.stdev, rank, world, args.seed*1000+step,
                              *part_paths(part)) for d, part, _ in sources]
    data = MixedDataset(datasets, [w for _, _, w in sources], args.seed*1000+step)
    dl = torch.utils.data.DataLoader(data, batch_size=args.batch, collate_fn=collate, num_workers=args.workers,
                                    prefetch_factor=4, persistent_workers=True, pin_memory=True,
                                    multiprocessing_context='spawn')
    iterator = iter(dl)
    val = {}
    for d, part, _ in sources:
        key = part if part not in val else f'{part}_{Path(d).name}'
        val[key] = fixed_windows(d, 'val', args.eval_n, model.encoder.mean, model.encoder.stdev, part)
    heldout = [] if args.no_heldout else fixed_windows(sources[0][0], 'heldout', args.heldout_n,
                                                       model.encoder.mean, model.encoder.stdev, sources[0][1])
    if heldout:
        assert len(heldout) == args.heldout_n, 'not enough heldout windows'
    import wandb
    wb = None
    if rank == 0:
        wb = wandb.init(project='viktor', name=out.name, id=run_id, resume='allow', dir=str(out),
                        mode=args.wandb_mode, config=vars(args) | {'world': world, 'global_batch': world*args.batch*args.accum})
        run_id = wb.id
        print(f'START {out.name} step={step} stop={stop} global_batch={world*args.batch*args.accum} '
              f'encoder={sum(p.numel() for p in model.encoder.base.parameters())} '
              f'compression={sum(p.numel() for p in model.encoder.stages.parameters())} '
              f'decoder={sum(p.numel() for p in model.decoder.parameters())}', flush=True)
        print('FlashAttention required in mature encoder and compression; decoder loss compiled=' + str(not args.no_compile), flush=True)
        (out/'config.json').write_text(json.dumps(vars(args), indent=2))
    dist.barrier()

    def save(predecay=False):
        if rank != 0:
            return
        state = {'step': step, 'args': vars(args), 'encoder': model.encoder.state_dict(),
                 'projector': model.projector.state_dict(), 'decoder': model.decoder.state_dict(),
                 'model': model.state_dict(), 'joint': joint, 'opt': opt.state_dict(), 'wandb_id': run_id}
        # Avoid storing duplicate module tensors in the resumable file.
        state.pop('model')
        torch.save(state, str(ckpt)+'.tmp')
        os.replace(str(ckpt)+'.tmp', ckpt)
        if predecay:
            import shutil
            shutil.copyfile(ckpt, out/f'predecay_{step}.pt')
        if step in (args.freeze_encoder_steps, 80000, args.decay_start, stop):
            torch.save({k:v for k,v in state.items() if k not in ('opt','wandb_id')}, out/f'snap_{step:07d}.pt')

    t0, sums, count = time.monotonic(), torch.zeros(6, device=dev, dtype=torch.float64), 0
    check = True
    while step < stop:
        if step == args.decay_start and not (out/f'predecay_{step}.pt').exists():
            save(predecay=True)
            dist.barrier()
        if not joint and step >= args.freeze_encoder_steps:
            del ddp
            torch._dynamo.reset()
            model.encoder.set_base_trainable(True)
            for group in named_groups(model.encoder.base, 'encoder', args.weight_decay):
                opt.add_param_group(group)
            joint = True
            ddp, check = wrap(), True
            if rank == 0:
                print(f'step {step}: Encoder unfrozen; rebuilt DDP/compiled loss; optimizer history preserved', flush=True)
        lr = wsd(step, total, args.lr, args.warmup, .2, args.decay_start)
        elr = lr / args.lr * args.encoder_lr * min(1., (step-args.freeze_encoder_steps+1)/args.encoder_warmup) if joint else 0.
        for group in opt.param_groups:
            group['lr'] = elr if group['name'] == 'encoder' else lr
        micros, denom = [], torch.zeros(2, device=dev, dtype=torch.float64)
        for _ in range(args.accum):
            sig, valid, seqs, ws, centers = next(iterator)
            ti, ta, weights, cs = model.batch(seqs, ws, args.pad_to, convert_centers(centers, args.compression))
            micros.append((sig, valid, ti, ta, weights, cs))
            denom += torch.tensor([float(weights.sum()), int((cs >= 0).sum())], device=dev)
        dist.all_reduce(denom)
        opt.zero_grad(set_to_none=True)
        for i, micro in enumerate(micros):
            context = ddp.no_sync() if i < args.accum-1 else contextlib.nullcontext()
            with context:
                sig, valid, ti, ta, weights, cs = (x.to(dev, non_blocking=True) for x in micro)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    _, ce, align, correct, scored = ddp(sig, valid, ti, ta, weights, cs)
                    weighted = ce * (world*weights.sum()/denom[0]) + .1*align*(world*(cs>=0).sum()/denom[1].clamp_min(1))
                if not torch.isfinite(weighted):
                    raise RuntimeError(f'nonfinite loss at step {step}')
                weighted.backward()
            sums += torch.stack([ce.detach().double()*weights.sum(), align.detach().double()*(cs>=0).sum(),
                                 correct.double(), scored.double(), weights.sum().double(), (cs>=0).sum().double()])
        watched = verify_gradients(model, opt, joint) if check else None
        norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip, error_if_nonfinite=True)
        opt.step()
        if watched is not None:
            verify_updates(watched, opt)
            check = False
        step, count = step+1, count+1
        if step % 20 == 0:
            temperature_guard(rank, dev, 78, 72)
        if step % args.log_every == 0:
            dist.all_reduce(sums)
            rec = {'train/ce': float(sums[0]/sums[4]), 'train/align': float(sums[1]/sums[5].clamp_min(1)),
                   'train/acc': float(sums[2]/sums[3]), 'lr': lr, 'encoder_lr': elr,
                   'grad_norm': float(norm), 's_per_step': (time.monotonic()-t0)/count,
                   'memory_gib': torch.cuda.max_memory_allocated()/2**30, 'joint': int(joint)}
            if rank == 0:
                wb.log(rec, step=step)
                print(f'step {step} ce {rec["train/ce"]:.4f} align {rec["train/align"]:.4f} '
                      f'acc {rec["train/acc"]:.4f} lr {lr:.3g} enc_lr {elr:.3g} '
                      f'{rec["s_per_step"]:.3f} s/step memory {rec["memory_gib"]:.2f} GiB', flush=True)
            sums.zero_()
            t0, count = time.monotonic(), 0
        if step % args.val_every == 0 or step == stop:
            stats = evaluate(model, val, args.eval_batch, dev)
            if rank == 0:
                wb.log(stats, step=step)
                print('VALIDATION '+str(step)+' '+json.dumps(stats), flush=True)
            t0, count = time.monotonic(), 0
        if step % args.ckpt_every == 0 or step in (args.freeze_encoder_steps, stop):
            save()
            dist.barrier()
            if heldout:
                model.eval()
                calls = sharded_calls(lambda w: heldout_calls(model, w, args.eval_batch, dev) if w else [], heldout)
                metrics = identity_stats(calls, heldout)
                identity = np.array([1-__import__('edlib').align(c, s, mode='NW', task='distance')['editDistance']/len(s)
                                     for c, (_,s,_) in zip(calls,heldout)])
                metrics['identity_p10'] = float(np.percentile(identity,10))
                metrics['length_ratio_p99'] = float(np.percentile([len(c)/len(s) for c,(_,s,_) in zip(calls,heldout)],99))
                if rank == 0:
                    wb.log({'heldout/'+k:v for k,v in metrics.items()}, step=step)
                    (out/f'heldout_{step:07d}.json').write_text(json.dumps(metrics,indent=2))
                    print('HELDOUT '+str(step)+' '+json.dumps(metrics),flush=True)
                model.train()
            t0, count = time.monotonic(), 0
    if rank == 0:
        wb.finish()
        print(f'COMPLETE step {step}',flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

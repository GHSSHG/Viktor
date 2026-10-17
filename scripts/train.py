"""Stage-1 SFT: frozen Dorado SUP encoder (koi forward, no gradients), single-base decoder + projector trained
from scratch on make_windows.py windows, one epoch of the train split. DDP via torchrun; decoder in fp32 with
bf16 autocast, compiled with torch.compile (label lengths padded to multiples of --pad-to).

Loss: weighted token cross-entropy + --align-weight x attention alignment loss (model.AlignConfig: chosen heads'
attention over the signal prefix pulled towards each base's forced-alignment frame; training only).
Training and validation windows exclude low-quality reads (data.split_rows); the chr20 heldout set keeps them.

Schedule: WSD (linear warmup, constant, 1 - sqrt decay to 0 over the last --decay-frac of the steps).
Logged to wandb (rank 0) every --log-every steps, including each GPU's temperature / SM clock / power.
  val      every --val-every steps: teacher-forced weighted loss / base accuracy on fixed 'val' windows
  heldout  at every checkpoint (--ckpt-every): free greedy decoding (DecodeEngine on a bf16 copy) of fixed
           chr20 windows: global identity (edlib NW), identity over scored bases, derailed fraction (< 0.9).
           The Dorado SUP baseline on the same windows (koi encoder + CRF head + koi beam search) is computed
           once into OUT/dorado.json.
Checkpoints: OUT/last.pt (resumable, optimizer + wandb run id, resumed automatically) and weights-only
OUT/snap_<step>.pt. Temperature guard: when any GPU reaches --temp-pause degC all ranks wait until every GPU
is at or below --temp-resume.

Example:
  torchrun --nproc-per-node 4 -m scripts.train --model-dir MODEL --windows ~/Work/BasecallerData/viktor_win12288 --out runs/sft1
"""

import argparse
import copy
import json
import math
import os
import time

import edlib
import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import torch.nn as nn

from viktor import Basecaller, DecodeEngine, get_tokenizer, load_dorado_encoder
from viktor.model import AlignConfig
from viktor.data import WindowDataset, collate, index_files, load_windows, split_rows


class Trainable(nn.Module):
    """The trained part (projector + decoder) for DDP; forward = Basecaller.weighted_loss."""

    def __init__(self, model):
        super().__init__()
        self.projector, self.decoder = model.projector, model.decoder
        self._model = [model]

    def forward(self, frames, tokens_in, targets, w, centers):
        return self._model[0].weighted_loss(frames, tokens_in, targets, w, centers)


def wsd(step, total, peak, warmup, decay_frac):
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:
        return peak * (step + 1) / warmup
    if step < decay_start:
        return peak
    return peak * (1 - math.sqrt((step - decay_start) / max(1, total - decay_start)))


def gpu_stats():
    """[(temperature degC, SM clock MHz, power W)] of every visible GPU (NVML via torch.cuda)."""
    return [(torch.cuda.temperature(i), torch.cuda.clock_rate(i), torch.cuda.power_draw(i) / 1000)
            for i in range(torch.cuda.device_count())]


def temperature_guard(rank, dev, pause, resume, wait=10):
    """All ranks: if any GPU is at or above `pause` degC, sleep until every GPU is at or below `resume`.
    Returns the seconds spent here."""
    t, waiting = time.time(), False
    while True:
        hot = torch.zeros(1, device=dev)
        if rank == 0:
            top = max(s[0] for s in gpu_stats())
            hot[0] = float(top > resume) if waiting else float(top >= pause)
            if hot.item() and not waiting:
                print(f"temperature guard: {top:.0f} degC, pausing until <= {resume:.0f}", flush=True)
        dist.broadcast(hot, 0)
        if hot.item() == 0:
            return time.time() - t
        waiting = True
        time.sleep(wait)


def masked_identity(pred, ref, scored):
    """Identity over scored reference bases (as SUPEncoder eval_free9): NW path; a mismatch or deletion counts
    against its reference base, an insertion against the reference base before it."""
    cig = edlib.align(pred, ref, mode="NW", task="path")["cigar"]
    ri, err, num = 0, 0, ""
    for ch in cig:
        if ch.isdigit():
            num += ch
            continue
        n, num = int(num), ""
        if ch == "=":
            ri += n
        elif ch in "XD":
            err += int(scored[ri:ri + n].sum())
            ri += n
        elif ch == "I":
            err += n * int(scored[max(ri - 1, 0)])
    return 1 - err / max(int(scored.sum()), 1)


def identity_stats(preds, wins):
    ident = np.array([1 - edlib.align(p, s, mode="NW", task="distance")["editDistance"] / len(s)
                      for p, (_, s, _) in zip(preds, wins)])
    masked = np.array([masked_identity(p, s, w > 0) for p, (_, s, w) in zip(preds, wins)])
    return {"identity_mean": float(ident.mean()), "identity_median": float(np.median(ident)),
            "derail": float((ident < 0.9).mean()), "masked_mean": float(masked.mean()),
            "masked_median": float(np.median(masked))}


def fixed_windows(index_dir, split, n, mean, stdev):
    """n windows of one split, sampled deterministically from a few index files."""
    files = index_files(index_dir)
    pick = np.random.default_rng(1).choice(len(files), min(8, len(files)), replace=False)
    idx = pl.concat([split_rows(pl.read_parquet(files[i]), split) for i in pick])
    return load_windows(idx.sample(min(n, idx.height), seed=1).to_dicts(), mean, stdev)


@torch.no_grad()
def dorado_calls(model_dir, encoder, wins, dev):
    import tomllib

    from scripts.basecall_crf import CRFHead, beam_search
    head = CRFHead(model_dir).to(dev).eval()
    with open(os.path.join(model_dir, "config.toml"), "rb") as f:
        q = tomllib.load(f)["qscore"]
    calls = []
    for i in range(0, len(wins), 256):
        x = torch.from_numpy(np.stack([w[0] for w in wins[i:i + 256]])).to(dev)[:, None]
        calls += [c[0] for c in beam_search(head(encoder(x)), q["scale"], q["bias"])]
    return calls


@torch.no_grad()
def eval_val(model, val, batch, dev):
    model.eval()
    loss = correct = scored = 0.0
    for i in range(0, len(val), batch):
        g = val[i:i + batch]
        sig, seqs, ws = collate(g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l, _, _, c, s = model.frame_loss(model.encoder(sig.to(dev)), seqs, ws)
        loss, correct, scored = loss + float(l) * len(g), correct + float(c), scored + float(s)
    model.train()
    return {"val/loss": loss / len(val), "val/acc": correct / scored}


@torch.no_grad()
def eval_heldout(model, heldout, batch, dev):
    """Free greedy decoding on a bf16 copy (the fused inference kernels need bf16 weights)."""
    inf = Basecaller(model.encoder, model.tokenizer, model.cfg)
    inf.projector = copy.deepcopy(model.projector).to(torch.bfloat16).eval()
    inf.decoder = copy.deepcopy(model.decoder).to(torch.bfloat16).eval()
    eng = DecodeEngine(inf, batch, max(len(s) for _, s, _ in heldout) + 64,
                       model.encoder.chunksize // model.encoder.stride)
    preds = []
    for i in range(0, len(heldout), batch):
        g = heldout[i:i + batch]
        g = g + [g[-1]] * (batch - len(g))
        frames = model.encoder(collate(g)[0].to(dev))
        toks = eng.generate(frames.reshape(-1, frames.shape[-1]), [frames.shape[1]] * batch).cpu()
        preds += [model.tokenizer.decode(t) for t in toks]
    del eng, inf
    torch.cuda.empty_cache()
    return {f"heldout/{k}": v for k, v in identity_stats(preds[:len(heldout)], heldout).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--windows", required=True, help="make_windows.py output dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32, help="windows per GPU per step")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests); default one epoch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--decay-frac", type=float, default=0.2, help="WSD: fraction of steps in the final decay")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--pad-to", type=int, default=128, help="pad label lengths to a multiple (compile shapes)")
    ap.add_argument("--align-weight", type=float, default=0.1, help="attention alignment loss weight (0 = off)")
    ap.add_argument("--align-layers", default="2,4", help="decoder layers with supervised heads")
    ap.add_argument("--align-heads", default="0,4", help="supervised query heads in each of those layers")
    ap.add_argument("--align-sigma", type=float, default=1.0, help="alignment target width, encoder frames")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="data loader workers per GPU")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=10_000, help="checkpoint + heldout evaluation")
    ap.add_argument("--eval-n", type=int, default=1024)
    ap.add_argument("--temp-every", type=int, default=20)
    ap.add_argument("--temp-pause", type=float, default=78)
    ap.add_argument("--temp-resume", type=float, default=72)
    ap.add_argument("--wandb-project", default="viktor")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(dev)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 64
    os.makedirs(args.out, exist_ok=True)

    n_train = split_rows(pl.scan_parquet(index_files(args.windows)), "train").select(pl.len()).collect().item()
    total = n_train // (args.batch * world)  # one epoch
    stop = min(total, args.max_steps or total)

    encoder = load_dorado_encoder(args.model_dir).to(dev, torch.float16).eval().use_koi()
    model = Basecaller(encoder, get_tokenizer("base"))
    model.set_encoder_trainable(False)
    if args.align_weight > 0:
        model.align = AlignConfig(tuple(int(x) for x in args.align_layers.split(",")),
                                  tuple(int(x) for x in args.align_heads.split(",")), args.align_sigma, args.align_weight)
    model.projector.to(dev)
    model.decoder.to(dev)
    trainable = Trainable(model)
    decay = [p for n, p in trainable.named_parameters() if p.ndim == 2 and "embed" not in n]
    no_decay = [p for n, p in trainable.named_parameters() if not (p.ndim == 2 and "embed" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95), fused=True)
    step, run_id = 0, None
    ckpt = os.path.join(args.out, "last.pt")
    if os.path.exists(ckpt):
        s = torch.load(ckpt, map_location=dev, weights_only=False)
        model.projector.load_state_dict(s["projector"])
        model.decoder.load_state_dict(s["decoder"])
        opt.load_state_dict(s["opt"])
        step, run_id = s["step"], s.get("wandb_id")
    ddp = nn.parallel.DistributedDataParallel(trainable, device_ids=[dev.index])
    step_fn = ddp if args.no_compile else torch.compile(ddp, dynamic=False)  # one graph per --pad-to length

    ds = WindowDataset(args.windows, "train", encoder.mean, encoder.stdev, rank, world, seed=args.seed * 1000 + step)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, collate_fn=collate, num_workers=args.workers,
                                     prefetch_factor=4, persistent_workers=True, pin_memory=True,
                                     multiprocessing_context="spawn")  # polars/pod5 threads do not survive fork
    wb = val = heldout = None
    if rank == 0:
        import wandb
        n_par = sum(p.numel() for p in trainable.parameters())
        wb = wandb.init(project=args.wandb_project, name=os.path.basename(os.path.normpath(args.out)), id=run_id,
                        resume="allow", dir=args.out,
                        config=vars(args) | {"world": world, "train_windows": n_train, "epoch_steps": total,
                                             "trainable_params": n_par})
        run_id = wb.id
        print(f"world {world}, {n_par / 1e6:.2f}M trainable, {n_train} train windows = {total} steps/epoch, "
              f"start at {step}, stop at {stop}", flush=True)
        val = fixed_windows(args.windows, "val", args.eval_n, encoder.mean, encoder.stdev)
        heldout = fixed_windows(args.windows, "heldout", args.eval_n, encoder.mean, encoder.stdev)
        base = os.path.join(args.out, "dorado.json")
        if not os.path.exists(base):
            with open(base, "w") as f:
                json.dump(identity_stats(dorado_calls(args.model_dir, encoder, heldout, dev), heldout), f)
        with open(base) as f:
            wb.summary.update({f"dorado/{k}": v for k, v in json.load(f).items()})
    dist.barrier()

    def save():
        state = {"step": step, "projector": model.projector.state_dict(), "decoder": model.decoder.state_dict(),
                 "opt": opt.state_dict(), "args": vars(args), "wandb_id": run_id}
        torch.save(state, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)
        torch.save({k: state[k] for k in ("step", "projector", "decoder", "args")},
                   os.path.join(args.out, f"snap_{step:07d}.pt"))

    t0, paused, acc = time.time(), 0.0, torch.zeros(6, device=dev)  # loss, ce, align sums, correct, scored, bases
    for sig, seqs, ws, cs in dl:
        if step >= stop:
            break
        lr = wsd(step, total, args.lr, args.warmup, args.decay_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        tokens_in, targets, w, centers = (t.to(dev, non_blocking=True) for t in model.batch(seqs, ws, args.pad_to, cs))
        with torch.no_grad():
            frames = encoder(sig.to(dev, non_blocking=True))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, ce, align, correct, scored = step_fn(frames, tokens_in, targets, w, centers)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(trainable.parameters(), args.clip)
        opt.step()
        step += 1
        acc += torch.stack([loss.detach(), ce.detach(), align.detach(), correct.float(), scored.float(),
                            torch.tensor(float(sum(map(len, seqs))), device=dev)])
        if step % args.temp_every == 0:
            paused += temperature_guard(rank, dev, args.temp_pause, args.temp_resume)
        if step % args.log_every == 0 or step == stop:
            dist.all_reduce(acc)
            n = args.log_every if step % args.log_every == 0 else step % args.log_every
            dt = time.time() - t0
            if rank == 0:
                rec = {"train/loss": float(acc[0]) / (n * world), "train/ce": float(acc[1]) / (n * world),
                       "train/align": float(acc[2]) / (n * world), "train/acc": float(acc[3] / acc[4]), "lr": lr,
                       "grad_norm": float(gnorm), "s_per_step": dt / n, "bases_per_s": float(acc[5]) / dt,
                       "paused_s": paused, "mem_gib": torch.cuda.max_memory_allocated() / 2**30}
                for i, (temp, clock, power) in enumerate(gpu_stats()):
                    rec |= {f"gpu{i}/temp": temp, f"gpu{i}/sm_mhz": clock, f"gpu{i}/power_w": power}
                wb.log(rec, step=step)
                print(f"step {step} loss {rec['train/loss']:.4f} ce {rec['train/ce']:.4f} align {rec['train/align']:.3f} acc {rec['train/acc']:.4f} lr {lr:.2e} "
                      f"{rec['s_per_step']:.3f} s/step {rec['bases_per_s']:.0f} bases/s", flush=True)
            t0, paused, acc = time.time(), 0.0, torch.zeros(6, device=dev)
        if step % args.val_every == 0 and rank == 0:
            wb.log(eval_val(model, val, 64, dev), step=step)
        if step % args.ckpt_every == 0 or step == stop:
            if rank == 0:
                save()
                wb.log(eval_heldout(model, heldout, 256, dev), step=step)
            dist.barrier()
            t0 = time.time()
    if rank == 0:
        wb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

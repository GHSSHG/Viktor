"""GSPO (group sequence policy optimisation) on top of an SFT checkpoint, data-parallel over torchrun ranks.

pool   greedy-decode --pool-scan random 'train' windows (w == 1) of the --windows sources (DIR[:PART[:WEIGHT]] as
       train.py; the scan is shared by weight) with the SFT model and keep a window pool: every hard window
       (identity < --hard-ident; a homopolymer the model cannot stop on already scores low) plus random others
       (--easy-frac of the pool); signals, labels, base weights and source part are stored in OUT/pool.pt. Each rank
       scans a share of the index files; rank 0 merges the shares.
train  every step: --windows-per-step pool windows (hard with probability 1 - --easy-frac), --group samples each
       (DecodeEngine, temperature --temperature) -> reward = masked identity to the reference label (errors on
       weight-0 bases are not counted; a call that never ends is scored as it stands). Advantage = reward
       standardised within the group; groups whose samples all score the same carry no signal and are dropped.
       Loss (GSPO, Zheng et al. 2025): sequence ratio s = exp(mean over the sequence's tokens of log pi - log pi_old),
       -min(s A, clip(s, 1 - eps, 1 + eps) A) averaged over sequences, plus --sft-weight x the weighted cross-entropy
       of the reference labels on the step's windows (keeps the policy near the labels). The rollout is split into
       --updates minibatches, one optimiser step each; pi_old = the policy before the first. Every rank rolls out its
       own --windows-per-step windows; gradients are averaged over ranks before each optimiser step (by hand: the
       number of live groups differs per rank), so the replicas stay identical.
       Evaluation (rank 0, every --eval-every steps, and at the start): greedy decoding of fixed windows of each
       --eval source (DIR:PART:SPLIT, sampled as train.py's heldout) -> <PART>_<SPLIT>/identity mean / median /
       derail, and the same over the windows whose label has a homopolymer >= --hard-hp. Temperature guard (all
       ranks) and checkpoints as train.py (OUT/last.pt resumes).

Examples (BD = ~/Work/BasecallerData):
  torchrun --nproc-per-node 4 -m scripts.gspo pool --model-dir MODEL --init runs/sft3/snap_1020000.pt --out runs/gspo2 \
      --windows $BD/viktor_win12288_part2:part2:0.85 $BD/viktor_win12288_bact:bact:0.15
  torchrun --nproc-per-node 4 -m scripts.gspo train --model-dir MODEL --init runs/sft3/snap_1020000.pt --out runs/gspo2 \
      --eval $BD/viktor_win12288_part2:part2:heldout $BD/viktor_win12288_bact:bact:val
"""

import argparse
import copy
import datetime
import os
import re
import time
from multiprocessing import get_context

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import torch.nn as nn

from scripts.train import fixed_windows, identity_stats, masked_identity, parse_source, temperature_guard
from viktor import Basecaller, DecodeEngine, get_tokenizer, load_dorado_encoder
from viktor.data import index_files, load_windows, part_paths, split_rows
from viktor.tokenizer import EOS, PAD

STEP_TOKENS = 2048  # rollout length cap (labels are at most ~1,600 bases)


def max_homopolymer(s):
    return max((len(m.group(0)) for m in re.finditer(r"(.)\1*", s)), default=0)


def load_model(model_dir, init, dev):
    encoder = load_dorado_encoder(model_dir).to(dev, torch.float16).eval().use_koi()
    model = Basecaller(encoder, get_tokenizer("base"))
    model.set_encoder_trainable(False)
    s = torch.load(init, map_location="cpu", weights_only=False)
    model.projector.load_state_dict(s["projector"])
    model.decoder.load_state_dict(s["decoder"])
    model.projector.to(dev)
    model.decoder.to(dev)
    return model


def bf16_copy(model):
    """Inference model sharing the encoder, with bf16 copies of the projector and decoder (fused kernels)."""
    inf = Basecaller(model.encoder, model.tokenizer, model.cfg)
    inf.projector = copy.deepcopy(model.projector).to(torch.bfloat16).eval()
    inf.decoder = copy.deepcopy(model.decoder).to(torch.bfloat16).eval()
    return inf


@torch.no_grad()
def sync(inf, model):
    """Copy the trained weights into the inference copy in place (the engines' CUDA graphs keep their pointers)."""
    for dst, src in zip(list(inf.projector.parameters()) + list(inf.decoder.parameters()),
                        list(model.projector.parameters()) + list(model.decoder.parameters())):
        dst.copy_(src)


@torch.no_grad()
def decode(engine, frames, tokenizer):
    """frames [N, P, d_enc] (N = engine batch) -> (calls, ended)."""
    toks = engine.generate(frames.reshape(-1, frames.shape[-1]), [frames.shape[1]] * frames.shape[0]).cpu()
    return [tokenizer.decode(t) for t in toks], (toks == EOS).any(1).tolist()


def _load_file(args):
    rows, mean, stdev, part = args
    return [w + (part,) for w in load_windows(rows, mean, stdev, part_paths(part)[0])]


def build_pool(args, dev, rank, world):
    model = load_model(args.model_dir, args.init, dev)
    rng = np.random.default_rng(args.seed + rank)
    sources = [parse_source(s) for s in args.windows]
    total_w = sum(w for _, _, w in sources)
    jobs = []
    for d, part, w in sources:
        files = index_files(d)
        per = -(-int(args.pool_scan * w / total_w) // len(files))
        for i, f in enumerate(files):
            idx = split_rows(pl.read_parquet(f), "train")
            jobs.append((idx.sample(min(per, idx.height), seed=args.seed + i).to_dicts(), model.encoder.mean,
                         model.encoder.stdev, part))
    jobs = jobs[rank::world]
    t = time.time()
    with get_context("spawn").Pool(max(1, 16 // world)) as pool:
        wins = [w for ws in pool.imap_unordered(_load_file, jobs) for w in ws]
    print(f"rank {rank}: {len(wins)} windows loaded, {time.time() - t:.0f} s", flush=True)
    eng = DecodeEngine(bf16_copy(model), args.batch, STEP_TOKENS, model.encoder.chunksize // model.encoder.stride)
    ident = np.empty(len(wins))
    for k, i in enumerate(range(0, len(wins), args.batch)):
        g = wins[i:i + args.batch]
        sig = torch.from_numpy(np.stack([w[0] for w in g + [g[-1]] * (args.batch - len(g))])).to(dev)
        with torch.no_grad():
            calls, _ = decode(eng, model.encoder(sig[:, None]), model.tokenizer)
        ident[i:i + len(g)] = [masked_identity(c, s, w > 0) for c, (_, s, w, _) in zip(calls, g)]
        if rank == 0 and (k + 1) % 20 == 0:
            done = ident[:i + len(g)]
            print(f"rank 0: decoded {i + len(g)} / {len(wins)}, {time.time() - t:.0f} s: identity mean {done.mean():.4f}, "
                  f"< {args.hard_ident}: {(done < args.hard_ident).mean():.4f}", flush=True)
    hp = np.array([max_homopolymer(w[1]) for w in wins])
    src = np.array([w[3] for w in wins])
    hard = ident < args.hard_ident
    easy = np.nonzero(~hard)[0]
    n_easy = min(len(easy), int(hard.sum() * args.easy_frac / (1 - args.easy_frac)))
    keep = np.concatenate([np.nonzero(hard)[0], rng.choice(easy, n_easy, replace=False)])
    share = os.path.join(args.out, f"pool_rank{rank}.pt")
    torch.save({"sig": torch.from_numpy(np.stack([wins[i][0] for i in keep])), "seq": [wins[i][1] for i in keep],
                "w": [wins[i][2] for i in keep], "ident": ident[keep], "hp": hp[keep], "hard": hard[keep],
                "src": src[keep].tolist()}, share)
    for part in sorted(set(src)):
        m = src == part
        print(f"rank {rank} {part}: scanned {int(m.sum())}: identity mean {ident[m].mean():.4f}, derail "
              f"{(ident[m] < 0.9).mean():.4f}, hard (identity < {args.hard_ident}) {int(hard[m].sum())}, "
              f"homopolymer >= {args.hard_hp}: {int((hp[m] >= args.hard_hp).sum())}, {time.time() - t:.0f} s", flush=True)
    del wins
    dist.barrier()
    if rank == 0:
        shares = [torch.load(os.path.join(args.out, f"pool_rank{r}.pt"), weights_only=False) for r in range(world)]
        pool = {k: (torch.cat([s[k] for s in shares]) if k == "sig" else
                    np.concatenate([s[k] for s in shares]) if isinstance(shares[0][k], np.ndarray) else
                    [x for s in shares for x in s[k]]) for k in shares[0]}
        torch.save(pool, os.path.join(args.out, "pool.pt"))
        for r in range(world):
            os.remove(os.path.join(args.out, f"pool_rank{r}.pt"))
        n = {p: (sum(x == p for x in pool["src"]), int(sum(h for x, h in zip(pool["src"], pool["hard"]) if x == p)))
             for p in sorted(set(pool["src"]))}
        print(f"pool {len(pool['seq'])} (hard {int(pool['hard'].sum())}); per part (windows, hard): {n}, "
              f"{time.time() - t:.0f} s", flush=True)
    dist.barrier()


def token_logprobs(model, frames, seqs, ended):
    """Teacher-forced log pi of each call's tokens [B, T] and the mask of real tokens (EOS only if the call ended)."""
    tokens_in, targets, _ = model.batch(seqs, pad_to=128)
    tokens_in, targets = tokens_in.to(frames.device), targets.to(frames.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model.logits(frames, tokens_in)
    lp = torch.log_softmax(logits.float(), -1).gather(-1, targets.clamp(min=0)[..., None])[..., 0]
    mask = targets != PAD
    for b, (s, e) in enumerate(zip(seqs, ended)):
        if not e:
            mask[b, len(s)] = False
    return lp, mask


def heldout_eval(engine, model, heldout, hp_min, name="heldout"):
    preds = []
    for i in range(0, len(heldout), engine.B):
        g = heldout[i:i + engine.B]
        sig = torch.from_numpy(np.stack([w[0] for w in g + [g[-1]] * (engine.B - len(g))])).to(engine.out.device)
        with torch.no_grad():
            preds += decode(engine, model.encoder(sig[:, None]), model.tokenizer)[0][:len(g)]
    rec = {f"{name}/{k}": v for k, v in identity_stats(preds, heldout).items()}
    sub = [i for i, w in enumerate(heldout) if max_homopolymer(w[1]) >= hp_min]
    if sub:
        rec |= {f"{name}_hp/{k}": v for k, v in identity_stats([preds[i] for i in sub], [heldout[i] for i in sub]).items()}
        rec[f"{name}_hp/windows"] = len(sub)
    return rec


def train(args, dev, rank, world):
    model = load_model(args.model_dir, args.init, dev)
    trainable = nn.ModuleList([model.projector, model.decoder])
    opt = torch.optim.AdamW(trainable.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    step, run_id = 0, None
    ckpt = os.path.join(args.out, "last.pt")
    if os.path.exists(ckpt):
        s = torch.load(ckpt, map_location=dev, weights_only=False)
        model.projector.load_state_dict(s["projector"])
        model.decoder.load_state_dict(s["decoder"])
        opt.load_state_dict(s["opt"])
        step, run_id = s["step"], s.get("wandb_id")
    pool = torch.load(os.path.join(args.out, "pool.pt"), weights_only=False)
    hard, easy = np.nonzero(pool["hard"])[0], np.nonzero(~pool["hard"])[0]
    inf = bf16_copy(model)
    B, G = args.windows_per_step, args.group
    roll = DecodeEngine(inf, B * G, STEP_TOKENS, model.encoder.chunksize // model.encoder.stride, temperature=args.temperature)
    trainable_params = list(trainable.parameters())
    wb = evals = greedy = None
    if rank == 0:
        evals = {}
        for spec in args.eval:
            d, part, split = spec.split(":")
            evals[f"{part}_{split}"] = fixed_windows(d, split, args.eval_n, model.encoder.mean, model.encoder.stdev, part)
        greedy = DecodeEngine(inf, args.eval_batch, max(len(s) for ws in evals.values() for _, s, _ in ws) + 64,
                              model.encoder.chunksize // model.encoder.stride)
        import wandb
        wb = wandb.init(project=args.wandb_project, name=os.path.basename(os.path.normpath(args.out)), id=run_id,
                        resume="allow", dir=args.out,
                        config=vars(args) | {"pool": len(pool["seq"]), "pool_hard": len(hard), "world": world})
    rng = np.random.default_rng(args.seed * 1000 + step + 7919 * rank)

    def evaluate():
        rec = {}
        if rank == 0:
            for name, ws in evals.items():
                rec |= heldout_eval(greedy, model, ws, args.hard_hp, name)
        dist.barrier()
        return rec

    def save():
        if rank == 0:
            state = {"step": step, "projector": model.projector.state_dict(), "decoder": model.decoder.state_dict(),
                     "opt": opt.state_dict(), "args": vars(args), "wandb_id": wb.id}
            torch.save(state, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
            torch.save({k: state[k] for k in ("step", "projector", "decoder", "args")},
                       os.path.join(args.out, f"snap_{step:07d}.pt"))
        dist.barrier()

    if step == 0:
        rec = evaluate()
        if rank == 0:
            wb.log(rec, step=0)
    while step < args.steps:
        t0 = time.time()
        paused = temperature_guard(rank, dev, args.temp_pause, args.temp_resume)
        n_hard = rng.binomial(B, 1 - args.easy_frac)
        pick = np.concatenate([rng.choice(hard, n_hard), rng.choice(easy, B - n_hard)])
        sig = pool["sig"][torch.from_numpy(pick)].to(dev)
        labels, weights = [pool["seq"][i] for i in pick], [pool["w"][i] for i in pick]
        with torch.no_grad():
            frames = model.encoder(sig[:, None])                                    # [B, P, d_enc]
        calls, ended = decode(roll, frames.repeat_interleave(G, 0), model.tokenizer)
        t_roll = time.time() - t0
        reward = np.array([masked_identity(c, labels[i // G], weights[i // G] > 0) for i, c in enumerate(calls)]).reshape(B, G)
        std = reward.std(1)
        live = np.nonzero(std > 1e-6)[0]                                            # groups with a learning signal
        adv = (reward - reward.mean(1, keepdims=True)) / (std[:, None] + 1e-6)
        seqs = [(b, j) for b in live for j in range(G)]
        # pi_old of every live sequence, before any update
        old = {}
        with torch.no_grad():
            for i in range(0, len(seqs), args.micro):
                mb = seqs[i:i + args.micro]
                lp, m = token_logprobs(model, frames[[b for b, _ in mb]], [calls[b * G + j] for b, j in mb],
                                       [ended[b * G + j] for b, j in mb])
                for k, bj in enumerate(mb):
                    old[bj] = lp[k][m[k]]
        stats = {"pg": 0.0, "ce": 0.0, "clip": 0.0, "ratio": 0.0, "n": 0}
        for u in range(args.updates):
            mine = [bj for bj in seqs if bj[0] % args.updates == u]
            wins = [b for b in range(B) if b % args.updates == u]
            opt.zero_grad(set_to_none=True)
            for i in range(0, len(mine), args.micro):
                mb = mine[i:i + args.micro]
                lp, m = token_logprobs(model, frames[[b for b, _ in mb]], [calls[b * G + j] for b, j in mb],
                                       [ended[b * G + j] for b, j in mb])
                s = torch.stack([(lp[k][m[k]] - old[bj]).mean().exp() for k, bj in enumerate(mb)])
                a = torch.tensor([adv[b, j] for b, j in mb], device=dev, dtype=torch.float32)
                pg = -torch.minimum(s * a, s.clamp(1 - args.eps, 1 + args.eps) * a)
                (pg.sum() / len(mine)).backward()
                stats["pg"] += float(pg.sum()); stats["n"] += len(mb)
                stats["clip"] += float(((s - 1).abs() > args.eps).sum()); stats["ratio"] += float((s - 1).abs().sum())
            for i in range(0, len(wins), args.micro):
                wb_ = wins[i:i + args.micro]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ce = model.frame_loss(frames[wb_], [labels[b] for b in wb_], [weights[b] for b in wb_])[1]
                (args.sft_weight * ce * len(wb_) / len(wins)).backward()
                stats["ce"] += float(ce) * len(wb_)
            # average over ranks; a rank without live groups contributes its CE gradient only
            for p in trainable_params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
                p.grad /= world
            gnorm = nn.utils.clip_grad_norm_(trainable_params, args.clip)
            opt.step()
        sync(inf, model)
        step += 1
        # step statistics summed over ranks
        tot = torch.tensor([reward.sum(), reward.max(1).sum(), (reward < 0.9).sum(), reward.size, len(live), B,
                            sum(ended), len(ended), stats["pg"], stats["ce"], stats["clip"], stats["ratio"], stats["n"]],
                           device=dev, dtype=torch.float64)
        dist.all_reduce(tot)
        r_sum, best, der, n_r, n_live, n_win, n_end, n_calls, pg, ce, clip, ratio, n = tot.tolist()
        n = max(n, 1)
        rec = {"reward/mean": r_sum / n_r, "reward/best_of_group": best / n_win, "reward/derail": der / n_r,
               "groups_live": n_live / n_win, "ended": n_end / n_calls, "pg": pg / n, "ce": ce / n_win,
               "clip_frac": clip / n, "ratio_dev": ratio / n, "grad_norm": float(gnorm), "lr": args.lr,
               "s_rollout": t_roll, "s_per_step": time.time() - t0 - paused, "paused_s": paused,
               "gpu/temp": torch.cuda.temperature(dev)}
        if step % args.eval_every == 0 or step == args.steps:
            rec |= evaluate()
            save()
        if rank == 0:
            wb.log(rec, step=step)
            if step % 10 == 0:
                print(f"step {step} reward {rec['reward/mean']:.4f} live {rec['groups_live']:.2f} pg {rec['pg']:.4f} "
                      f"ce {rec['ce']:.4f} clip {rec['clip_frac']:.3f} {rec['s_per_step']:.1f} s/step", flush=True)
    if rank == 0:
        wb.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pool", "train"])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--init", required=True, help="SFT checkpoint (snap_*.pt or last.pt)")
    bd = os.path.expanduser("~/Work/BasecallerData")
    ap.add_argument("--windows", nargs="+", default=[f"{bd}/viktor_win12288_part2:part2:0.85", f"{bd}/viktor_win12288_bact:bact:0.15"],
                    help="pool: window index sources DIR[:PART[:WEIGHT]]")
    ap.add_argument("--eval", nargs="+", default=[f"{bd}/viktor_win12288_part2:part2:heldout", f"{bd}/viktor_win12288_bact:bact:val"],
                    help="train: evaluation sources DIR:PART:SPLIT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool-scan", type=int, default=200_000, help="pool: train windows decoded")
    ap.add_argument("--batch", type=int, default=512, help="pool: decode batch")
    ap.add_argument("--hard-ident", type=float, default=0.99)
    ap.add_argument("--hard-hp", type=int, default=8, help="homopolymer length of the heldout_hp evaluation subset")
    ap.add_argument("--easy-frac", type=float, default=0.3, help="share of easy windows in the pool and in each step")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--windows-per-step", type=int, default=64)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--eps", type=float, default=4e-4, help="GSPO sequence-ratio clip")
    ap.add_argument("--updates", type=int, default=2, help="optimiser steps per rollout")
    ap.add_argument("--sft-weight", type=float, default=0.05)
    ap.add_argument("--micro", type=int, default=32, help="sequences per forward/backward")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=1024)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--temp-pause", type=float, default=80)
    ap.add_argument("--temp-resume", type=float, default=72)
    ap.add_argument("--wandb-project", default="viktor")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(dev)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    (build_pool if args.cmd == "pool" else train)(args, dev, rank, world)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""GSPO (group sequence policy optimisation) on top of an SFT checkpoint, data-parallel over torchrun ranks.

scan   greedy-decode 'train' windows (w == 1) of the --windows sources (DIR[:PART[:WEIGHT]] as train.py) with the
       SFT model until --pool-scan windows are scanned in all (shared by weight, then evenly over each source's index
       files). Each index file's train rows are taken in a fixed shuffled order, so a larger --pool-scan later (or a rerun
       after a crash) only decodes the rows not yet in the scan table (OUT/scan/*.parquet: part, file, row, masked identity,
       longest label homopolymer, label length, call, ended; one table per --init, checked by OUT/scan/init.txt).
dorado Dorado SUP (CRF head, beam search) on the scanned windows with identity < --hard-ident that the cache
       (--dorado-cache, shared by all runs: Dorado and the labels do not change) does not have yet. Per window:
       Dorado's masked identity, the call-to-call identity, and 'shared' = the share of the model's errors (scored
       reference positions) that Dorado also makes within +-2 bases. A window both get wrong in the same places is
       more likely a label problem (the two share the Dorado encoder, so agreement is evidence, not proof).
pool   (one process) classes from the scan table: derail (identity < --derail-ident), hard (< --hard-ident), easy;
       drops label suspects (model and Dorado identity < --suspect-ident and shared >= --suspect-shared), keeps up
       to --pool-max windows of each class (random) and stores signals, labels, base weights, class and source part
       in OUT/pool.pt.
train  every step: --windows-per-step pool windows drawn by class shares --mix (derail,hard,easy), --group samples each
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

Examples (defaults: --windows part1 / part2 / bact / hall at 0.539 / 0.161 / 0.0795 / 0.2205, --eval part2 heldout,
bact val, hall val):
  torchrun --nproc-per-node 3 -m scripts.gspo scan --model-dir MODEL --init runs/sft4/snap_1540000.pt --out runs/gspo3 \
      --pool-scan 1000000
  torchrun --nproc-per-node 3 -m scripts.gspo dorado --model-dir MODEL --init runs/sft4/snap_1540000.pt --out runs/gspo3
  torchrun --nproc-per-node 1 -m scripts.gspo pool --model-dir MODEL --init runs/sft4/snap_1540000.pt --out runs/gspo3
  torchrun --nproc-per-node 3 -m scripts.gspo train --model-dir MODEL --init runs/sft4/snap_1540000.pt --out runs/gspo3
"""

import argparse
import copy
import datetime
import glob
import os
import re
import time
import zlib
from multiprocessing import get_context

import edlib
import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import torch.nn as nn

from scripts.train import (dorado_calls, fixed_windows, identity_stats, masked_identity, parse_source, ref_errors,
                           temperature_guard)
from viktor import Basecaller, DecodeEngine, get_tokenizer, load_dorado_encoder
from viktor.data import index_files, load_windows, part_paths, split_rows
from viktor.tokenizer import EOS, PAD

STEP_TOKENS = 2048  # rollout length cap (labels are at most ~1,600 bases)
CLASSES = ("derail", "hard", "easy")
KEY = ["part", "file", "row"]  # a window: index file (basename) and row number in it


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


def _load_rows(job):
    """(part, file, index rows, mean, stdev) -> the same plus their (signal, label, weights) windows."""
    part, file, rows, mean, stdev = job
    return part, file, rows, load_windows(rows, mean, stdev, part_paths(part)[0])


def stream(jobs, procs, ahead=32):
    """_load_rows results in job order, at most `ahead` jobs loaded ahead of the consumer."""
    with get_context("spawn").Pool(procs) as pool:
        for i in range(0, len(jobs), ahead):
            yield from pool.imap(_load_rows, jobs[i:i + ahead])


def read_table(d):
    files = sorted(glob.glob(os.path.join(d, "*.parquet")))
    return pl.read_parquet(files) if files else None


def write_table(d, rank, rec):
    if rec[KEY[0]]:
        pl.DataFrame(rec).write_parquet(os.path.join(d, f"r{rank}_{time.time_ns()}.parquet"))
        for v in rec.values():
            v.clear()


def index_rows(index_dir, file, rows=None):
    """One index file's train rows (with their row number) in the file's fixed scan order, or the given rows."""
    idx = pl.read_parquet(os.path.join(index_dir, file)).with_row_index("row").with_columns(pl.col("row").cast(pl.Int64))
    if rows is not None:
        return idx.filter(pl.col("row").is_in(rows))
    return split_rows(idx, "train").sample(fraction=1.0, shuffle=True, seed=zlib.crc32(file.encode()))


def chunks(part, file, idx, mean, stdev, size=1000):
    rows = idx.to_dicts()
    return [(part, file, rows[i:i + size], mean, stdev) for i in range(0, len(rows), size)]


def scan(args, dev, rank, world):
    out = os.path.join(args.out, "scan")
    os.makedirs(out, exist_ok=True)
    tag = os.path.join(out, "init.txt")
    if rank == 0 and not os.path.exists(tag):
        with open(tag, "w") as f:
            f.write(os.path.abspath(args.init))
    dist.barrier()
    with open(tag) as f:
        assert f.read() == os.path.abspath(args.init), f"{out} holds a scan of another --init"
    model = load_model(args.model_dir, args.init, dev)
    done = read_table(out)
    have = {} if done is None else {(p, f): r for p, f, r in done.group_by("part", "file").agg("row").iter_rows()}
    sources = [parse_source(s) for s in args.windows]
    total_w = sum(w for _, _, w in sources)
    jobs = []
    for d, part, w in sources:
        files = [os.path.basename(f) for f in index_files(d)]
        per = -(-int(args.pool_scan * w / total_w) // len(files))
        for f in files:
            if len(have.get((part, f), [])) < per:  # the first `per` rows of the scan order not yet in the table
                todo = index_rows(d, f)[:per].filter(~pl.col("row").is_in(have.get((part, f), [])))
                jobs += chunks(part, f, todo, model.encoder.mean, model.encoder.stdev)
    jobs = jobs[rank::world]
    n_todo = sum(len(j[2]) for j in jobs)
    eng = DecodeEngine(bf16_copy(model), args.batch, STEP_TOKENS, model.encoder.chunksize // model.encoder.stride)
    rec = {k: [] for k in KEY + ["ident", "hp", "n_bases", "call", "ended"]}
    buf, n, t = [], 0, time.time()

    def flush(g):
        sig = torch.from_numpy(np.stack([x[3][0] for x in g + [g[-1]] * (args.batch - len(g))])).to(dev)
        with torch.no_grad():
            calls, ended = decode(eng, model.encoder(sig[:, None]), model.tokenizer)
        for (part, file, row, (_, seq, w)), c, e in zip(g, calls, ended):
            for k, v in zip(rec, (part, file, row, masked_identity(c, seq, w > 0), max_homopolymer(seq), len(seq), c, e)):
                rec[k].append(v)

    for part, file, rows, wins in stream(jobs, max(1, 16 // world)):
        buf += [(part, file, r["row"], x) for r, x in zip(rows, wins)]
        while len(buf) >= args.batch or (buf and n + len(buf) == n_todo):
            g, buf = buf[:args.batch], buf[args.batch:]
            flush(g)
            n += len(g)
            if len(rec["row"]) >= 50_000:
                write_table(out, rank, rec)
            if rank == 0 and n // args.batch % 20 == 0:
                print(f"rank 0: scanned {n} / {n_todo}, {time.time() - t:.0f} s", flush=True)
    write_table(out, rank, rec)
    dist.barrier()
    if rank == 0:
        sc = read_table(out)
        summary = sc.group_by("part").agg(pl.len().alias("windows"), pl.col("ident").mean().alias("identity_mean"),
                                          (pl.col("ident") < args.derail_ident).mean().alias("derail"),
                                          (pl.col("ident") < args.hard_ident).mean().alias("hard")).sort("part")
        print(f"scan table: {sc.height} windows ({time.time() - t:.0f} s)\n{summary}", flush=True)


def shared_errors(a, b, tol=2):
    """Share of the positions a (sorted) with a position of b (sorted) within tol."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    i = np.searchsorted(b, a)
    lo, hi = b[np.clip(i - 1, 0, len(b) - 1)], b[np.clip(i, 0, len(b) - 1)]
    return float(((np.abs(a - lo) <= tol) | (np.abs(a - hi) <= tol)).mean())


def dorado_check(args, dev, rank, world):
    sc = read_table(os.path.join(args.out, "scan")).filter(pl.col("ident") < args.hard_ident)
    cache = read_table(args.dorado_cache)
    if cache is not None:
        sc = sc.join(cache.select(KEY), on=KEY, how="anti")
    os.makedirs(args.dorado_cache, exist_ok=True)
    dirs = {part: d for d, part, _ in (parse_source(s) for s in args.windows)}
    encoder = load_dorado_encoder(args.model_dir).to(dev, torch.float16).eval().use_koi()
    jobs = []
    for (part, file), g in sc.sort(KEY).group_by(["part", "file"], maintain_order=True):
        idx = index_rows(dirs[part], file, g["row"]).join(g.select("row", "call"), on="row", how="left")
        jobs += chunks(part, file, idx, encoder.mean, encoder.stdev)
    jobs = jobs[rank::world]
    n_todo = sum(len(j[2]) for j in jobs)
    rec = {k: [] for k in KEY + ["d_ident", "vd_ident", "shared", "v_err", "d_err"]}
    n, t = 0, time.time()
    for part, file, rows, wins in stream(jobs, max(1, 16 // world)):
        for r, (_, seq, w), d in zip(rows, wins, dorado_calls(args.model_dir, encoder, wins, dev)):
            scored = w > 0
            ev, ed = ref_errors(r["call"], seq), ref_errors(d, seq)
            ev, ed = ev[scored[ev]], ed[scored[ed]]
            vd = edlib.align(r["call"], d, mode="NW", task="distance")["editDistance"]
            for k, v in zip(rec, (part, file, r["row"], 1 - len(ed) / max(int(scored.sum()), 1), 1 - vd / max(len(d), 1),
                                  shared_errors(np.unique(ev), np.unique(ed)), len(ev), len(ed))):
                rec[k].append(v)
        n += len(rows)
        if len(rec["row"]) >= 50_000:
            write_table(args.dorado_cache, rank, rec)
        if rank == 0:
            print(f"rank 0: Dorado {n} / {n_todo}, {time.time() - t:.0f} s", flush=True)
    write_table(args.dorado_cache, rank, rec)
    dist.barrier()


def build_pool(args):
    t = time.time()
    ident = pl.col("ident")
    sc = read_table(os.path.join(args.out, "scan")).drop("call").with_columns(
        cls=pl.when(ident < args.derail_ident).then(0).when(ident < args.hard_ident).then(1).otherwise(2))
    cache = read_table(args.dorado_cache)
    if cache is None:
        sc = sc.with_columns(d_ident=pl.lit(None, pl.Float64), suspect=pl.lit(False))
    else:
        sc = sc.join(cache, on=KEY, how="left").with_columns(
            suspect=((ident < args.suspect_ident) & (pl.col("d_ident") < args.suspect_ident)
                     & (pl.col("shared") >= args.suspect_shared)).fill_null(False))
    print(sc.group_by("part", "cls").agg(pl.len().alias("windows"), pl.col("suspect").sum().alias("suspect"),
                                         (pl.col("d_ident").is_null() & (pl.col("cls") < 2)).sum().alias("no_dorado"))
          .sort("part", "cls"), flush=True)
    keep = sc.filter(~pl.col("suspect"))
    pick = pl.concat([keep.filter(pl.col("cls") == c).sample(min(args.pool_max, int((keep["cls"] == c).sum())), seed=args.seed)
                      for c in range(len(CLASSES))]).sort(KEY)
    dirs = {part: d for d, part, _ in (parse_source(s) for s in args.windows)}
    encoder = load_dorado_encoder(args.model_dir)
    jobs = []
    for (part, file), g in pick.group_by(["part", "file"], maintain_order=True):
        idx = index_rows(dirs[part], file, g["row"]).join(g.select("row", "ident", "hp", "cls"), on="row", how="left")
        jobs += chunks(part, file, idx, encoder.mean, encoder.stdev)
    pool = {"sig": [], "seq": [], "w": [], "ident": [], "hp": [], "cls": [], "src": []}
    for part, _, rows, wins in stream(jobs, 32):
        for r, (x, seq, w) in zip(rows, wins):
            for k, v in zip(pool, (x, seq, w, r["ident"], r["hp"], r["cls"], part)):
                pool[k].append(v)
    pool["sig"] = torch.from_numpy(np.stack(pool["sig"]))
    for k in ("ident", "hp", "cls"):
        pool[k] = np.array(pool[k])
    torch.save(pool, os.path.join(args.out, "pool.pt"))
    n = {p: [int(((np.array(pool["src"]) == p) & (pool["cls"] == c)).sum()) for c in range(len(CLASSES))]
         for p in sorted(set(pool["src"]))}
    print(f"pool {len(pool['seq'])}; per part {'/'.join(CLASSES)}: {n}, {time.time() - t:.0f} s", flush=True)


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
    by_cls = [np.nonzero(pool["cls"] == c)[0] for c in range(len(CLASSES))]
    mix = np.array([float(x) for x in args.mix.split(",")])
    mix /= mix.sum()
    assert all(len(ix) for ix, m in zip(by_cls, mix) if m > 0), [len(ix) for ix in by_cls]
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
                        config=vars(args) | {"pool": len(pool["seq"]), "world": world}
                        | {f"pool_{c}": len(ix) for c, ix in zip(CLASSES, by_cls)})
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
        pick = np.concatenate([rng.choice(ix, k) for ix, k in zip(by_cls, rng.multinomial(B, mix)) if k])
        cls = pool["cls"][pick]
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
        per_cls = [x for c in range(len(CLASSES)) for x in (reward[cls == c].sum(), reward[cls == c].size)]
        tot = torch.tensor([reward.sum(), reward.max(1).sum(), (reward < 0.9).sum(), reward.size, len(live), B,
                            sum(ended), len(ended), stats["pg"], stats["ce"], stats["clip"], stats["ratio"], stats["n"]]
                           + per_cls, device=dev, dtype=torch.float64)
        dist.all_reduce(tot)
        r_sum, best, der, n_r, n_live, n_win, n_end, n_calls, pg, ce, clip, ratio, n = tot.tolist()[:13]
        per_cls = tot.tolist()[13:]
        n = max(n, 1)
        rec = {f"reward_cls/{c}": per_cls[2 * i] / per_cls[2 * i + 1] for i, c in enumerate(CLASSES) if per_cls[2 * i + 1]}
        rec |= {"reward/mean": r_sum / n_r, "reward/best_of_group": best / n_win, "reward/derail": der / n_r,
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
    ap.add_argument("cmd", choices=["scan", "dorado", "pool", "train"])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--init", required=True, help="SFT checkpoint (snap_*.pt or last.pt)")
    bd = os.path.expanduser("~/Work/BasecallerData")
    # human 0.7 (part1 : part2 by train windows ~77 : 23) + bacteria 0.3 (Sowpati : Hall by train windows)
    ap.add_argument("--windows", nargs="+", default=[f"{bd}/viktor_win12288:part1:0.539", f"{bd}/viktor_win12288_part2:part2:0.161",
                                                     f"{bd}/viktor_win12288_bact:bact:0.0795", f"{bd}/viktor_win12288_hall:hall:0.2205"],
                    help="scan / dorado / pool: window index sources DIR[:PART[:WEIGHT]]")
    ap.add_argument("--eval", nargs="+", default=[f"{bd}/viktor_win12288_part2:part2:heldout", f"{bd}/viktor_win12288_bact:bact:val",
                                                  f"{bd}/viktor_win12288_hall:hall:val"],
                    help="train: evaluation sources DIR:PART:SPLIT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool-scan", type=int, default=1_000_000, help="scan: train windows in the scan table")
    ap.add_argument("--batch", type=int, default=512, help="scan: decode batch")
    ap.add_argument("--dorado-cache", default="runs/dorado_windows", help="dorado / pool: Dorado check table (all runs)")
    ap.add_argument("--derail-ident", type=float, default=0.9)
    ap.add_argument("--hard-ident", type=float, default=0.99)
    ap.add_argument("--suspect-ident", type=float, default=0.97, help="pool: label suspect if model and Dorado are below")
    ap.add_argument("--suspect-shared", type=float, default=0.8, help="pool: ... and Dorado shares this much of the errors")
    ap.add_argument("--pool-max", type=int, default=100_000, help="pool: windows kept per class")
    ap.add_argument("--hard-hp", type=int, default=8, help="homopolymer length of the heldout_hp evaluation subset")
    ap.add_argument("--mix", default="0.15,0.55,0.30", help="train: shares of derail,hard,easy windows per step")
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
    if args.cmd == "pool":
        assert world == 1, "pool runs in one process"
        build_pool(args)
    else:
        {"scan": scan, "dorado": dorado_check, "train": train}[args.cmd](args, dev, rank, world)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

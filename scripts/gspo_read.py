"""Read-level GSPO: the reward of a sampled chunk call is scored after stitching it into its read, as basecall_viktor does.

A training entry is one chunk k of a read (Dorado chunking as basecall_viktor: 12288 samples, overlap 600, the short
last chunk anchored to the read end) together with its neighbours k-1 and k+1. Every step the neighbours are decoded
greedily with the current policy (shared by the group), --group candidates of chunk k are sampled, and each candidate
is stitched with its neighbours exactly as at inference (homopolymer cap, overlap_cut at the overlap middle; adapter
trim on a read's first chunk). Only chunk k's own stitched part is scored: the first ANCHOR bases of the right
neighbour and the last ANCHOR bases of the left neighbour beyond the cuts are placed in the truth (edlib HW), which
fixes the truth span of the own part, and the own part is aligned end to end into that span; so duplicated / dropped
seam bases, a derailed, truncated or empty chunk all cost reward, while the neighbours' errors do not. At a read end
the truth span ends at the segment's first / last labelled base (prep: 'lo' / 'hi'): every labelled base must be
called, and only up to 'free_lo' / 'free_hi' own bases beyond it (adapter rest, barcode, unlabelled bases: estimated
from the unlabelled signal there, capped per sequencing kit by FREE_CAP) are free, the rest count as insertions. An entry whose inner side has no anchor (the
neighbour does not align) is not scored (NaN: skipped in mining, training and evaluation). Truth: the reference bases
of the read's stage-6 segment whose signal midpoints lie within chunk k's span +- --pad samples, extended by
--ref-margin bases of the reference on both sides. Reward = 1 - errors / max(scored own bases, truth span); errors on
masked label bases (index mask_lo / mask_hi, N) do not count.

prep   (one process, CPU) --reads train-split reads and --eval-reads val-split reads of the --sources (weights: share of
       the reads; default human : bacteria = 4 : 1, split within each as runs/k4.sh) with one stage-6 segment, at least
       two chunks and no id in --exclude-ids. Per read one chunk k: with probability --edge-share the first or last chunk
       (when the segment reaches within --edge-slack samples of that read end), else a middle chunk whose neighbours lie
       inside the segment. Stores the signal of chunks k-1..k+1 (OUT/readpool/<split>.bin, fp16) and per entry its spans,
       position (first / middle / last), truth and truth weights (OUT/readpool/<split>.pt), plus --sft-windows random
       train windows of the same sources for the SFT term (OUT/readpool/sft.pt).
train  (torchrun) GSPO as scripts/gspo.py train (sequence ratio per base, clip --eps, advantage over max(group std,
       --std-floor), dead groups dropped, --updates optimiser steps per rollout, gradients averaged over ranks by hand),
       encoder frozen; plus --sft-weight x the weighted cross-entropy of --sft-per-step replayed SFT windows; with
       --tok-weight > 0 plus a token credit term: in each group a candidate is right at a truth position when it has
       no error within --tok-window bases, the baseline there is the share of candidates right, every own base gets
       (right - baseline) at its truth base and a token the mean over its own bases; loss = PPO-clipped token ratio
       (--tok-eps) x credit, summed over the credited tokens / the call's tokens, x --tok-weight. Entries
       are drawn by class (--mix): first a mining pass scores the greedy call of every train entry under --init
       (readpool/mine.pt, reused while --init is unchanged) and splits human entries into low / mid / perfect. Evaluation
       (rank 0, every --eval-every steps and at the start): the read-level reward of greedy calls of the eval entries
       (read/<group>/... and read/<position>/...), the stratified --eval-windows set and the --eval window sources as
       scripts/gspo.py. Checkpoints as scripts/gspo.py (OUT/last.pt resumes; snapshots carry the tuned encoder).

Examples:
  python -m scripts.gspo_read prep --model-dir MODEL --out runs/gspo_read
  torchrun --nproc-per-node 4 -m scripts.gspo_read train --model-dir MODEL --init runs/k4ft2/snap_0700000.pt --out runs/gspo_read
"""

import argparse
import datetime
import os
import random
import time
import zlib
from multiprocessing import get_context

import edlib
import numpy as np
import pod5
import polars as pl
import pysam
import torch
import torch.distributed as dist
import torch.nn as nn

from scripts.basecall_viktor import cap_homopolymers, overlap_cut, trim_adapter
from scripts.evaluate import scores
from scripts.gspo import bf16_copy, chunks, decode, engine, load_model, lr_at, stream, sync, token_logprobs
from scripts.train import fixed_windows, heldout_calls, identity_stats, parse_source, temperature_guard
from viktor import load_dorado_encoder
from viktor.chunking import chunk_read, dorado_scale
from viktor.data import (CRF_FRAME, DNA_TRIM_START, SIZE, STRIDE, collate, index_files, load_windows, part_paths,
                         pod5_path, split_rows, win_len)

POSITIONS = ("first", "middle", "last")
HUMAN = ("part1", "part2", "part3")
COLS = ["read_id", "parent_id", "pod5", "split_start", "num_samples", "win", "win_len", "contig", "strand", "g_lo",
        "g_hi", "mask_lo", "mask_hi", "seg_off", "seg_n", "f_first", "f_exit", "ad_i0", "ad_i1", "label", "lw", "split", "w"]
_COMP = str.maketrans("ACGTN", "TGCAN")
ANCHOR = 150  # neighbour bases used to place the own part in the truth
FREE_X, FREE_PAD = 1.5, 30  # read ends: free own bases beyond the labelled truth (see _prep_file)
# ... capped per (group, sequencing kit) at (start, end) bases: p95 + ~10 of k4ft2 700k's output beyond its alignment
# (after trim_adapter, as the reward sees it) on the read-level test sets, 2026-11-08: human LSK114 0 / 11; bacteria
# LSK114 (lab primer) 86 / 95, LSK114-XL 0 / 21, NBD114-24/-96 47 / 56 (barcode), RBK114-96 (runs with native barcodes) 99 / 78
FREE_CAP = {("human", "sqk-lsk114"): (10, 20), ("bact", "sqk-lsk114"): (100, 105), ("bact", "sqk-lsk114-xl"): (10, 30),
            ("bact", "sqk-nbd114-24"): (55, 65), ("bact", "sqk-nbd114-96"): (55, 65), ("bact", "sqk-rbk114-96"): (110, 90)}
FREE_CAP_DEFAULT = (70, 70)


# ---------------------------------------------------------------------------------------------------------- reward

def local_sequence(calls, spans, t, margin):
    """calls / spans: the entry's chunk calls (chunk t = the candidate) and sample spans -> (left, kept, right): chunk
    t's kept span as stitch() keeps it and up to `margin` bases of each neighbour's kept part beyond the cuts."""
    s, start, end, left, right = calls[t], 0, len(calls[t]), "", ""
    if t > 0:
        a = calls[t - 1]
        ca, start = overlap_cut(a, s, (spans[t - 1][1] - spans[t][0]) // STRIDE,
                                len(a) / max((spans[t - 1][1] - spans[t - 1][0]) // STRIDE, 1))
        left = a[:ca][-margin:]
    if t + 1 < len(calls):
        b = calls[t + 1]
        end, cb = overlap_cut(s, b, (spans[t][1] - spans[t + 1][0]) // STRIDE,
                              len(s) / max((spans[t][1] - spans[t][0]) // STRIDE, 1))
        right = b[cb:][:margin]
    return left, s[start:end], right, start


def flank_anchor(flank, truth, left):
    """Truth positions where the own part may start (left neighbour: ends of its last ANCHOR bases in truth) or end
    (right neighbour: starts of its first ANCHOR bases), every equally good placement (repeats); retried with
    ANCHOR // 3 bases (the flank may run past the truth end); [] when the flank is short or does not align (> 25% edits)."""
    if not flank:
        return []
    for n in (ANCHOR, ANCHOR // 3):
        part = flank[-n:] if left else flank[:n]
        if len(part) < ANCHOR // 6:
            return []
        r = edlib.align(part, truth, mode="HW", task="locations")
        if 0 <= r["editDistance"] <= 0.25 * len(part):
            return sorted({hi + 1 if left else lo for lo, hi in r["locations"]})
    return []


def align_pairs(own, t, kind):
    """own aligned to all of truth t: kind 'NW' end to end, 'free_start' / 'free_end' with own's start / end left open
    -> (alignment columns [(own index or -1, truth index or -1)] in order, own bases left open)."""
    if kind == "free_start":
        p, n = align_pairs(own[::-1], t[::-1], "free_end")
        return [(len(own) - 1 - i if i >= 0 else -1, len(t) - 1 - j if j >= 0 else -1) for i, j in reversed(p)], n
    if not t:
        return ([(i, -1) for i in range(len(own))], 0) if kind == "NW" else ([], len(own))
    if not own:
        return [(-1, j) for j in range(len(t))], 0
    if kind == "NW":
        r, ins = edlib.align(own, t, mode="NW", task="path"), "I"  # I: own base without truth base
    else:
        r, ins = edlib.align(t, own, mode="SHW", task="path"), "D"  # query = truth (whole), own's end free
    cols, i, j, num = [], 0, 0, ""
    for ch in r["cigar"]:
        if ch.isdigit():
            num += ch
            continue
        for _ in range(int(num)):
            if ch in "=X":
                cols.append((i, j)); i += 1; j += 1
            elif ch == ins:
                cols.append((i, -1)); i += 1
            else:
                cols.append((-1, j)); j += 1
        num = ""
    return cols, len(own) - i


def score_columns(cols, n_open, own, t, w, free):
    """-> (weighted errors, own bases scored, truth positions of the errors): a mismatch / missing truth base at its
    truth base, an extra own base at the truth base before it; errors at weight-0 truth bases do not count; open own
    bases beyond `free` count as insertions (no position)."""
    err, prev, pos = 0.0, 0, []
    for i, j in cols:
        if j >= 0:
            prev = j
            if i < 0 or own[i] != t[j]:
                err += float(w[j]); pos += [j] if w[j] else []
        elif len(t):
            err += float(w[prev]); pos += [prev] if w[prev] else []
        else:
            err += 1.0
    over = max(0, n_open - free)
    return err + over, sum(i >= 0 for i, _ in cols) + over, pos


def read_reward(own, truth, w, left="", right="", edge=None, detail=False):
    """1 - errors / max(scored own bases, truth bases of the own part). The own part's truth span is fixed on each side
    by the neighbour (flank_anchor of the left / right flank) or, at a read end, by the segment's first / last labelled
    base (edge: lo / hi with free_lo / free_hi own bases beyond it not counted), and every truth base of the span must
    be called, so dropped, truncated or duplicated bases all count. An inner side without an anchor (empty or unaligned
    neighbour) -> NaN (not scored). The neighbours' own errors cost nothing (the same for every candidate of a group).
    detail: -> (reward, (truth offset a, alignment columns, error truth positions)) for token credit."""
    edge = edge or {}
    lo, hi = edge.get("lo"), edge.get("hi")
    A = flank_anchor(left, truth, True) if lo is None else []
    B = flank_anchor(right, truth, False) if hi is None else []
    if (lo is None and not A) or (hi is None and not B):
        return (float("nan"), None) if detail else float("nan")
    L, R = A or [lo], B or [hi]
    # in repeats: the pair whose span is closest to the own length
    a, b = min(((x, max(x, y)) for x in L for y in R), key=lambda p: abs(p[1] - p[0] - len(own)))
    kind, free = ("free_start", edge["free_lo"]) if lo is not None else ("free_end", edge["free_hi"]) if hi is not None \
        else ("NW", 0)
    cols, n_open = align_pairs(own, truth[a:b], kind)
    err, n_own, pos = score_columns(cols, n_open, own, truth[a:b], w[a:b], free)
    r = max(0.0, 1 - err / max(n_own, b - a, 1))
    return (r, (a, cols, pos)) if detail else r


def entry_reward(calls, e, margin, cap, detail=False):
    """Calls of an entry's chunks (raw decoder output) -> its read-level reward (chunk t's own stitched part).
    detail: -> (reward, read_reward detail, raw call index of own base 0 or None when the cap changed the call)."""
    raw = calls[e["t"]]
    if cap:
        calls = [cap_homopolymers(c, cap) for c in calls]
    left, kept, right, off = local_sequence(calls, e["spans"], e["t"], margin)
    if e["pos"] == "first":
        trimmed = trim_adapter(kept)
        off, kept = off + len(kept) - len(trimmed), trimmed
    if not detail:
        return read_reward(kept, e["truth"], e["tw"], left, right, e.get("edge"))
    r, d = read_reward(kept, e["truth"], e["tw"], left, right, e.get("edge"), detail=True)
    return r, d, off if calls[e["t"]] == raw else None


def token_credit(details, k, win):
    """Per-position credit within a group. details: per candidate (read_reward detail, raw offset) or None. A candidate
    is right at truth position p when it has no error within +- win of p; the baseline at p is the share of the
    candidates covering p that are right there. Each own base gets (right - baseline) at its truth base (an extra base:
    the truth base before it); a token gets the mean over its own bases -> per candidate {token index: credit}."""
    use = [g for g, x in enumerate(details) if x is not None and x[0] is not None and x[1] is not None]
    out = [{} for _ in details]
    if len(use) < 2:
        return out
    maps = {}
    for g in use:
        (a, cols, pos), off = details[g]
        own_i, tp, js, prev = [], [], [], 0
        for i, j in cols:
            if j >= 0:
                prev = j; js.append(j)
            if i >= 0:
                own_i.append(i); tp.append(a + (j if j >= 0 else prev))
        if js:  # truth span of the candidate (every truth base of it is a column)
            maps[g] = (np.array(own_i, int), np.array(tp, int), a + np.array(pos, int), off, a + js[0], a + js[-1] + 1)
    if len(maps) < 2:
        return out
    use = list(maps)
    lo, hi = min(m[4] for m in maps.values()), max(m[5] for m in maps.values())
    n = hi - lo
    right = np.zeros((len(use), n)); cover = np.zeros((len(use), n), bool)
    for r_, g in enumerate(use):
        _, _, ep, _, s0, s1 = maps[g]
        cover[r_, s0 - lo:s1 - lo] = True
        bad = np.zeros(n + 2 * win)
        np.add.at(bad, ep - lo + win, 1)
        right[r_] = np.convolve(bad, np.ones(2 * win + 1), "same")[win:win + n] == 0
    base = (right * cover).sum(0) / np.maximum(cover.sum(0), 1)
    for r_, g in enumerate(use):
        own_i, tp, _, off, _, _ = maps[g]
        if not len(tp):
            continue
        c = right[r_, tp - lo] - base[tp - lo]
        tok = (off + own_i) // k
        s = np.bincount(tok, c); m = np.bincount(tok)
        out[g] = {int(t): float(s[t] / m[t]) for t in np.nonzero((m > 0) & (np.abs(s) > 1e-9))[0]}
    return out


def group_rewards(task):
    """(candidate calls of chunk t, the entry's neighbour calls by chunk index, entry, margin, cap[, (k, win)]) ->
    rewards, or with (k, win): (rewards, per candidate token credit)."""
    cands, nb, e, margin, cap = task[:5]
    calls = lambda c: [c if i == e["t"] else nb[i] for i in range(len(e["spans"]))]
    if len(task) < 6 or task[5] is None:
        return [entry_reward(calls(c), e, margin, cap) for c in cands]
    res = [entry_reward(calls(c), e, margin, cap, detail=True) for c in cands]
    rewards = [r for r, _, _ in res]
    if any(np.isnan(rewards)):
        return rewards, [{} for _ in cands]
    return rewards, token_credit([(d, off) for _, d, off in res], *task[5])


# ------------------------------------------------------------------------------------------------------------ prep

def segment_midpoints(flat, off, n, f0, fx):
    """Signal midpoint of each base of a stage-6 segment, in samples of the scaled signal (DNA_TRIM_START removed)."""
    e = np.empty(n + 1, np.int64)
    e[0], e[n] = f0, fx + 1
    e[1:n] = f0 + np.cumsum(flat[off:off + n - 1], dtype=np.int64)
    return CRF_FRAME // 2 * (e[:-1] + e[1:]) - DNA_TRIM_START


def _prep_file(job):
    """One index file's chosen reads -> [(entry, chunk signal)]."""
    src, part, d, file, reads, P = job
    fa = pysam.FastaFile(part_paths(part)[0])
    flat = np.load(os.path.join(d, "dwell", os.path.basename(file)[:-len(".parquet")] + ".npy"), mmap_mode="r")
    recs, out = {}, []
    with pod5.Reader(pod5_path(reads[0][0]["pod5"])) as reader:
        for rec in reader.reads(selection=list({rows[0]["parent_id"] for rows in reads}), missing_ok=True):
            recs[str(rec.read_id)] = (rec.signal, rec.calibration.offset, rec.calibration.scale,
                                      getattr(rec, "open_pore_level", float("nan")), rec.run_info.flow_cell_product_code,
                                      rec.run_info.sequencing_kit.lower())
    for rows in reads:
        r0 = rows[0]
        if r0["parent_id"] not in recs:
            continue
        raw, offset, scale, pore, flow_cell, kit = recs[r0["parent_id"]]
        cap_lo, cap_hi = FREE_CAP.get(("human" if part in HUMAN else "bact", kit), FREE_CAP_DEFAULT)
        x = dorado_scale(raw[r0["split_start"]: r0["split_start"] + r0["num_samples"]], offset, scale, P["mean"],
                         P["stdev"], pore, flow_cell)
        n_x = len(x)
        spans = chunk_read(n_x, SIZE, P["overlap"])
        if spans[-1][1] - spans[-1][0] < SIZE:  # basecall_viktor --anchor-tail
            spans[-1] = (n_x - SIZE, n_x)
        off, n, f0, fx = r0["seg_off"], r0["seg_n"], r0["f_first"], r0["f_exit"]
        mid = segment_midpoints(flat, off, n, f0, fx)
        # the segment's reference anchor (t_start for '+', t_end for '-'), from every reference-labelled window
        anchors, masked = set(), []
        for r in rows:
            if r.get("label") is not None:
                continue
            lo = int(np.searchsorted(mid, r["win"]))
            hi = int(np.searchsorted(mid, r["win"] + win_len(r)))
            if hi - lo != r["g_hi"] - r["g_lo"]:
                anchors.add(None)
                break
            anchors.add(r["g_lo"] - lo if r["strand"] == "+" else r["g_hi"] + lo)
            if r.get("lw") is None or not len(r["lw"]):
                nad = (r.get("ad_i1") or 0) - (r.get("ad_i0") or 0)
                masked += [(lo + a - nad, lo + b - nad) for a, b in zip(r["mask_lo"], r["mask_hi"])]
        if len(anchors) != 1 or None in anchors:
            continue
        anchor, strand, contig = anchors.pop(), r0["strand"], r0["contig"]
        s_lo, s_hi = CRF_FRAME * f0 - DNA_TRIM_START, CRF_FRAME * (fx + 1) - DNA_TRIM_START
        last = len(spans) - 1
        rng = random.Random(zlib.crc32(r0["read_id"].encode()) + P["seed"])
        middle = [k for k in range(1, last) if spans[k - 1][0] >= s_lo and spans[k + 1][1] <= s_hi]
        edges = ([0] if s_lo <= P["edge_slack"] and spans[1][1] <= s_hi else []) + \
                ([last] if n_x - s_hi <= P["edge_slack"] and spans[last - 1][0] >= s_lo else [])
        pick = edges if edges and (rng.random() < P["edge_share"] or not middle) else middle
        if not pick:
            continue
        k = rng.choice(pick)
        ks = list(range(max(k - 1, 0), min(k + 1, last) + 1))
        a0 = spans[ks[0]][0]
        if np.searchsorted(mid, spans[k][1]) - np.searchsorted(mid, spans[k][0]) < P["min_bases"]:
            continue  # chunk k carries almost no labelled bases (stall / segment gap): nothing to learn
        # truth: segment bases with midpoints in chunk k +- pad, +- ref_margin bases of reference (read order)
        i0 = int(np.searchsorted(mid, spans[k][0] - P["pad"])) - P["ref_margin"]
        i1 = int(np.searchsorted(mid, spans[k][1] + P["pad"])) + P["ref_margin"]
        clen = fa.get_reference_length(contig)
        if strand == "+":
            g0, g1 = max(anchor + i0, 0), min(anchor + i1, clen)
            truth, j0 = fa.fetch(contig, g0, g1).upper(), g0 - anchor
        else:
            g0, g1 = max(anchor - i1, 0), min(anchor - i0, clen)
            truth, j0 = fa.fetch(contig, g0, g1).upper().translate(_COMP)[::-1], anchor - g1
        tw = (np.frombuffer(truth.encode(), np.uint8) != ord("N")).astype(np.uint8)
        for a, b in masked:  # segment base indices -> truth indices
            tw[max(a - j0, 0):max(b - j0, 0)] = 0
        # read end: truth index of the segment's first / last labelled base (segment base i = truth index i - j0) and
        # the own bases allowed beyond it (adapter rest, barcode, unlabelled bases): FREE_X x the bases the unlabelled
        # signal there would carry at the segment's rate + FREE_PAD, at most the kit's FREE_CAP
        rate = n / max(s_hi - s_lo, 1)
        edge = {"lo": -j0, "free_lo": min(int(FREE_X * max(s_lo, 0) * rate) + FREE_PAD, cap_lo)} if k == 0 else \
               {"hi": n - j0, "free_hi": min(int(FREE_X * max(n_x - s_hi, 0) * rate) + FREE_PAD, cap_hi)} if k == last else None
        out.append(({"src": src, "part": part, "read_id": r0["read_id"],
                     "pos": "first" if k == 0 else "last" if k == last else "middle",
                     "spans": [(spans[j][0] - a0, spans[j][1] - a0) for j in ks], "t": ks.index(k),
                     "truth": truth, "tw": tw, "edge": edge}, x[a0:spans[ks[-1]][1]].copy()))
    return out


def pick_reads(srcs, split, n_reads, exclude, seed):
    """Per source (share of n_reads by weight) reads of one split with a single stage-6 segment and >= 2 chunks,
    spread over the index files -> jobs (source, part, index dir, index file, [rows of each read])."""
    total = sum(w for _, _, w in srcs)
    jobs, rng = [], random.Random(seed)
    for d, part, w in srcs:
        quota = round(n_reads * w / total)
        files = index_files(d)
        rng.shuffle(files)
        per = 2 * -(-quota // len(files))
        for f in files:
            if quota <= 0:
                break
            schema = pl.read_parquet_schema(f)
            idx = split_rows(pl.read_parquet(f, columns=[c for c in COLS if c in schema]), split).filter(
                pl.col("seg_off").is_not_null() & (pl.col("num_samples") > SIZE + DNA_TRIM_START)
                & ~pl.col("read_id").is_in(exclude) & ~pl.col("parent_id").is_in(exclude))
            want = min(per, quota)
            ids = idx["read_id"].unique().sort().to_list()
            rng.shuffle(ids)
            reads = []
            for rows in idx.filter(pl.col("read_id").is_in(ids[:3 * want])).partition_by("read_id"):
                if rows.select(pl.struct("seg_off", "seg_n", "f_first", "f_exit").n_unique()).item() == 1:
                    reads.append(rows.sort("win").to_dicts())
            reads = reads[:want]
            if reads:
                jobs.append((os.path.basename(os.path.normpath(d)), part, d, f, reads))
                quota -= len(reads)
        print(f"{split} {os.path.basename(os.path.normpath(d))}: {sum(len(j[4]) for j in jobs if j[2] == d)} reads",
              flush=True)
    return jobs


def prep(args):
    out = os.path.join(args.out, "readpool")
    os.makedirs(out, exist_ok=True)
    enc = load_dorado_encoder(args.model_dir)
    srcs = [parse_source(s) for s in args.sources]
    exclude = set()
    for p in args.exclude_ids:
        if os.path.exists(p):
            exclude |= {l.strip() for l in open(p) if l.strip()}
    P = {"mean": enc.mean, "stdev": enc.stdev, "overlap": enc.overlap, "edge_share": args.edge_share,
         "edge_slack": args.edge_slack, "pad": args.pad, "ref_margin": args.ref_margin, "seed": args.seed,
         "min_bases": args.min_bases}
    t = time.time()
    for split, n_reads in (("train", args.reads), ("val", args.eval_reads)):
        if os.path.exists(os.path.join(out, f"{split}.pt")):
            continue
        jobs = [j + (P,) for j in pick_reads(srcs, split, n_reads, list(exclude), args.seed)]
        entries, off = [], 0
        with open(os.path.join(out, f"{split}.bin.tmp"), "wb") as fb, get_context("spawn").Pool(args.procs) as pool:
            for res in pool.imap_unordered(_prep_file, jobs):
                for e, x in res:
                    e["off"] = off
                    fb.write(x.astype(np.float16).tobytes())
                    off += len(x)
                    entries.append(e)
                print(f"{split}: {len(entries)} entries, {time.time() - t:.0f} s", flush=True)
        os.replace(os.path.join(out, f"{split}.bin.tmp"), os.path.join(out, f"{split}.bin"))
        torch.save({"entries": entries, "samples": off}, os.path.join(out, f"{split}.pt"))
        cnt = {(g, p): sum((e["part"] in HUMAN) == (g == "human") and e["pos"] == p for e in entries)
               for g in ("human", "bact") for p in POSITIONS}
        print(f"{split}: {len(entries)} entries, {off * 2 / 1e9:.2f} GB signal; per group / position: {cnt}", flush=True)
    if not os.path.exists(os.path.join(out, "sft.pt")):
        total, jobs = sum(w for _, _, w in srcs), []
        rng = np.random.default_rng(args.seed)
        for d, part, w in srcs:
            files = index_files(d)
            pick = rng.choice(len(files), min(10, len(files)), replace=False)
            idx = pl.concat([split_rows(pl.read_parquet(files[i]), "train").with_columns(pl.lit(os.path.basename(files[i])).alias("_f"))
                             for i in pick], how="diagonal_relaxed")
            idx = idx.sample(min(round(args.sft_windows * w / total), idx.height), seed=args.seed)
            for (f,), g in idx.group_by("_f"):
                jobs += chunks(os.path.basename(os.path.normpath(d)), part, f, g.drop("_f"), enc.mean, enc.stdev)
        wins = [x for _, _, _, ws in stream(jobs, args.procs) for x in ws]
        sig, valid = collate(wins)[:2]
        torch.save({"sig": sig[:, 0], "valid": valid, "seq": [s for _, s, _ in wins], "w": [w for _, _, w in wins]},
                   os.path.join(out, "sft.pt"))
        print(f"sft: {len(wins)} windows, {time.time() - t:.0f} s", flush=True)


# ----------------------------------------------------------------------------------------------------------- train

class ReadPool:
    def __init__(self, d, split):
        meta = torch.load(os.path.join(d, f"{split}.pt"), weights_only=False)
        self.entries = meta["entries"]
        self.sig = np.memmap(os.path.join(d, f"{split}.bin"), np.float16, "r", shape=(meta["samples"],))

    def chunks(self, ix):
        """Entries ix -> (signals [M, 1, SIZE] fp16, per entry its chunks' rows in them)."""
        sigs, rows = [], []
        for i in ix:
            e = self.entries[i]
            rows.append(list(range(len(sigs), len(sigs) + len(e["spans"]))))
            sigs += [self.sig[e["off"] + a:e["off"] + b] for a, b in e["spans"]]
        return torch.from_numpy(np.stack(sigs))[:, None], rows


def run(eng, frames, plens, tokenizer):
    """decode() for any number of sequences (padded to the engine batch, in passes)."""
    calls, ended = [], []
    for i in range(0, len(frames), eng.B):
        f, p = frames[i:i + eng.B], plens[i:i + eng.B]
        n = len(f)
        if n < eng.B:
            f = torch.cat([f, f[-1:].expand(eng.B - n, -1, -1)])
            p = torch.cat([p, p[-1:].expand(eng.B - n)])
        c, e = decode(eng, f, p, tokenizer)
        calls += c[:n]
        ended += e[:n]
    return calls, ended


def encode(model, sig, dev):
    with torch.no_grad():
        return model.encode(sig.to(dev), torch.full((len(sig),), sig.shape[-1]))


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
    pdir = os.path.join(args.pool or args.out, "readpool")
    pool = ReadPool(pdir, "train")
    rpool = get_context("spawn").Pool(args.reward_procs)  # rewards: stitching + alignment on the CPU, one task per group
    sft = torch.load(os.path.join(pdir, "sft.pt"), weights_only=False)
    inf = bf16_copy(model)
    B, G = args.reads_per_step, args.group
    roll = engine(inf, B * G, args.temperature)
    greedy = engine(inf, 2 * B)
    trainable_params = list(trainable.parameters())
    wb = evals = v3 = val = None
    if rank == 0:
        val = ReadPool(pdir, "val")
        evals = {}
        for spec in args.eval:
            d, part, split = spec.split(":")
            evals[f"{part}_{split}"] = fixed_windows(d, split, args.eval_n, model.encoder.mean, model.encoder.stdev, part)
        v3_rows = pl.read_parquet(args.eval_windows).to_dicts()
        v3 = v3_rows, load_windows(v3_rows, model.encoder.mean, model.encoder.stdev, args.eval_ref)
        import wandb
        wb = wandb.init(project=args.wandb_project, name=os.path.basename(os.path.normpath(args.out)), id=run_id,
                        resume="allow", dir=args.out, config=vars(args) | {"pool": len(pool.entries), "world": world})
    rng = np.random.default_rng(args.seed * 1000 + step + 7919 * rank)
    group = lambda e: "human" if e["part"] in HUMAN else "bact"

    def greedy_rewards(rp, ix, batch):
        """Read-level rewards of the greedy calls of entries ix of pool rp."""
        eng = engine(inf, batch)
        rewards = []
        per = batch // 3
        for i in range(0, len(ix), per):
            sub = ix[i:i + per]
            sig, rows = rp.chunks(sub)
            frames, plens = encode(model, sig, dev)
            calls, _ = run(eng, frames, plens, model.tokenizer)
            rewards += [x[0] for x in rpool.map(group_rewards, [([calls[r[rp.entries[b]["t"]]]], {k: calls[j] for k, j in enumerate(r)},
                                                                rp.entries[b], args.margin, args.homopolymer_cap)
                                                               for r, b in zip(rows, sub)])]
        del eng
        torch.cuda.empty_cache()
        return np.array(rewards)

    # mining: greedy reward of every train entry under the initial policy -> sampling classes (cached per --init)
    mine_path = os.path.join(pdir, "mine.pt")
    m = torch.load(mine_path, weights_only=False) if os.path.exists(mine_path) else None
    key = os.path.abspath(args.mine_init or args.init)
    if m is None or m["init"] != key or len(m["reward"]) != len(pool.entries):
        t = time.time()
        part = list(range(rank, len(pool.entries), world))
        got = [None] * world
        dist.all_gather_object(got, (part, greedy_rewards(pool, part, args.mine_batch).tolist()))
        rew = np.zeros(len(pool.entries))
        for p, r in got:
            rew[p] = r
        m = {"init": key, "reward": rew}
        if rank == 0:
            torch.save(m, mine_path)
            print(f"mined {len(rew)} entries in {time.time() - t:.0f} s", flush=True)
        dist.barrier()
    mix = {k: float(v) for k, v in (x.split(":") for x in args.mix.split(","))}
    cls = np.array(["skip" if np.isnan(r) else "bact" if group(e) == "bact" else "low" if r < args.low else
                    "mid" if r < args.perfect else "perfect" for e, r in zip(pool.entries, m["reward"])])
    members = {k: np.nonzero(cls == k)[0] for k in mix}
    mix = {k: v for k, v in mix.items() if len(members[k])}
    classes, probs = list(mix), np.array(list(mix.values())) / sum(mix.values())
    if rank == 0:
        print("pool classes: " + ", ".join(f"{k} {len(members[k])} (sampled {p:.2f}, greedy mean {m['reward'][members[k]].mean():.4f})"
                                         for k, p in zip(classes, probs)) + f"; not scored {int((cls == 'skip').sum())}", flush=True)

    def read_eval():
        """Greedy calls of every chunk of the eval entries -> read-level reward statistics."""
        rewards = greedy_rewards(val, list(range(len(val.entries))), args.eval_batch)
        ok = ~np.isnan(rewards)
        rec = {"read/not_scored": int((~ok).sum())}
        for name, sel in [("all", ok)] + \
                         [(g, ok & np.array([group(e) == g for e in val.entries])) for g in ("human", "bact")] + \
                         [(p, ok & np.array([e["pos"] == p for e in val.entries])) for p in POSITIONS]:
            if sel.any():
                r = rewards[sel]
                rec |= {f"read/{name}/mean": float(r.mean()), f"read/{name}/median": float(np.median(r)),
                        f"read/{name}/derail": float((r < 0.9).mean())}
        return rec

    def evaluate():
        rec = {}
        if rank == 0:
            rec |= read_eval()
            rows, wins = v3
            r = scores(heldout_calls(model, wins, args.eval_batch, dev), wins, rows)
            rec |= {f"v3/{st}/{k}": v for st, d in r.items() for k, v in d.items() if k != "n"}
            for name, ws in evals.items():
                rec |= {f"{name}/{k}": v for k, v in identity_stats(heldout_calls(model, ws, args.eval_batch, dev), ws).items()}
            mmd = lambda g: "/".join(f"{100 * rec.get(f'read/{g}/{k}', 0):.2f}" for k in ("mean", "median", "derail"))
            print(f"eval step {step}: read mean/median/derail human {mmd('human')} | bact {mmd('bact')} | first "
                  f"{mmd('first')} middle {mmd('middle')} last {mmd('last')} (not scored {rec['read/not_scored']}); v3 normal trim mean/derail "
                  f"{100 * r['normal']['trim_mean']:.2f}/{100 * r['normal']['trim_derail']:.2f}", flush=True)
        dist.barrier()
        return rec

    def save():
        if rank == 0:
            state = {"step": step, "projector": model.projector.state_dict(), "decoder": model.decoder.state_dict(),
                     "opt": opt.state_dict(), "args": vars(args) | {"tokenizer": "4mer" if model.tokenizer.k == 4 else "base",
                                                          "layers": model.cfg.n_layers},
                     "wandb_id": wb.id}
            if model.tuned_encoder is not None:
                state["encoder"] = model.tuned_encoder
            torch.save(state, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
            torch.save({k: state[k] for k in ("step", "projector", "decoder", "args", "encoder") if k in state},
                       os.path.join(args.out, f"snap_{step:07d}.pt"))
        dist.barrier()

    if step == 0:
        rec = evaluate()
        if rank == 0:
            wb.log(rec, step=0)
    while step < args.steps:
        t0 = time.time()
        paused = temperature_guard(rank, dev, args.temp_pause, args.temp_resume)
        lr = lr_at(args, step)
        for g in opt.param_groups:
            g["lr"] = lr
        pick = np.array([rng.choice(members[c]) for c in rng.choice(classes, B, p=probs)])
        ents = [pool.entries[i] for i in pick]
        sig, rows = pool.chunks(pick)
        frames, plens = encode(model, sig, dev)
        tgt = torch.tensor([r[e["t"]] for r, e in zip(rows, ents)], device=dev)
        nb = [j for r, e in zip(rows, ents) for i, j in enumerate(r) if i != e["t"]]   # neighbours: greedy, shared by the group
        nb_calls = dict(zip(nb, run(greedy, frames[nb], plens[nb], model.tokenizer)[0])) if nb else {}
        tf, tp = frames[tgt], plens[tgt]
        calls, ended = decode(roll, tf.repeat_interleave(G, 0), tp.repeat_interleave(G), model.tokenizer)
        t_roll = time.time() - t0
        credit_args = (model.tokenizer.k, args.tok_window) if args.tok_weight > 0 else None
        tasks = [(calls[b * G:(b + 1) * G], {i: nb_calls[c] for i, c in enumerate(rows[b]) if i != e["t"]}, e,
                  args.margin, args.homopolymer_cap, credit_args) for b, e in enumerate(ents)]
        res = rpool.map(group_rewards, tasks)
        if credit_args:
            reward, credit = np.array([r for r, _ in res]), [c for _, c in res]
        else:
            reward, credit = np.array(res), [[{}] * G for _ in res]
        t_rew = time.time() - t0 - t_roll
        ok = ~np.isnan(reward).any(1)  # a group with an unscored candidate (neighbour does not align) is dropped
        reward[~ok] = 0.0
        std = reward.std(1)
        has_credit = np.array([any(c) for c in credit])
        live = np.nonzero(ok & ((std > 1e-6) | has_credit))[0]
        adv = (reward - reward.mean(1, keepdims=True)) / np.maximum(std, args.std_floor)[:, None]
        seqs = [(b, j) for b in live for j in range(G)]
        n_bases = {bj: max(len(calls[bj[0] * G + bj[1]]), 1) for bj in seqs}
        tok_adv = {}  # (b, j) -> (token indices, credits) on the device
        for bj in seqs:
            c = credit[bj[0]][bj[1]]
            if c:
                tok_adv[bj] = (torch.tensor(list(c), device=dev), torch.tensor(list(c.values()), device=dev, dtype=torch.float32))

        def logprobs(mb):
            ix = [b for b, _ in mb]
            return token_logprobs(model, tf[ix], tp[ix], [calls[b * G + j] for b, j in mb], [ended[b * G + j] for b, j in mb])

        old = {}
        with torch.no_grad():
            for i in range(0, len(seqs), args.micro):
                mb = seqs[i:i + args.micro]
                lp, m = logprobs(mb)
                for k, bj in enumerate(mb):
                    old[bj] = lp[k][m[k]]
        sft_pick = torch.from_numpy(rng.choice(len(sft["seq"]), args.sft_per_step))
        sft_frames, sft_plens = model.encode(sft["sig"][sft_pick][:, None].to(dev), sft["valid"][sft_pick])
        stats = {"pg": 0.0, "ce": 0.0, "clip": 0.0, "ratio": 0.0, "n": 0, "tok": 0.0, "tok_n": 0, "tok_clip": 0.0, "tokens": 0, "tok_abs": 0.0}
        for u in range(args.updates):
            mine = [bj for bj in seqs if bj[0] % args.updates == u]
            wins = [b for b in range(len(sft_pick)) if b % args.updates == u]
            opt.zero_grad(set_to_none=True)
            for i in range(0, len(mine), args.micro):
                mb = mine[i:i + args.micro]
                lp, m = logprobs(mb)
                lps = [lp[k][m[k]] for k in range(len(mb))]
                s = torch.stack([((x - old[bj]).sum() / n_bases[bj]).exp() for x, bj in zip(lps, mb)])
                a = torch.tensor([adv[b, j] for b, j in mb], device=dev, dtype=torch.float32)
                pg = -torch.minimum(s * a, s.clamp(1 - args.eps, 1 + args.eps) * a)
                loss = pg.sum()
                # token term: PPO-clipped token ratio x position credit, summed over the credited tokens / the call's tokens
                for x, bj in zip(lps, mb):
                    stats["tokens"] += len(x)
                    if bj not in tok_adv:
                        continue
                    idx, c = tok_adv[bj]
                    keep = idx < len(x)
                    idx, c = idx[keep], c[keep]
                    rt = (x[idx] - old[bj][idx]).exp()
                    lt = -torch.minimum(rt * c, rt.clamp(1 - args.tok_eps, 1 + args.tok_eps) * c).sum() / len(x)
                    loss = loss + args.tok_weight * lt
                    stats["tok"] += float(lt.detach()); stats["tok_n"] += len(idx)
                    stats["tok_clip"] += float(((rt.detach() - 1).abs() > args.tok_eps).sum()); stats["tok_abs"] += float(c.abs().sum())
                (loss / len(mine)).backward()
                stats["pg"] += float(pg.detach().sum()); stats["n"] += len(mb)
                stats["clip"] += float(((s - 1).abs() > args.eps).sum()); stats["ratio"] += float((s.detach() - 1).abs().sum())
            for i in range(0, len(wins), args.micro):
                wb_ = wins[i:i + args.micro]
                ix = sft_pick[wb_].tolist()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ce = model.frame_loss(sft_frames[wb_], [sft["seq"][k] for k in ix], [sft["w"][k] for k in ix],
                                          plens=sft_plens[wb_])[1]
                (args.sft_weight * ce * len(wb_) / len(wins)).backward()
                stats["ce"] += float(ce) * len(wb_)
            for p in trainable_params:  # average over ranks (a rank without live groups contributes its CE gradient only)
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
                p.grad /= world
            gnorm = nn.utils.clip_grad_norm_(trainable_params, args.clip)
            opt.step()
        sync(inf, model)
        step += 1
        keys = [("group", g) for g in ("human", "bact")] + [("pos", p) for p in POSITIONS]
        sel = lambda kind, v, b, e: ok[b] and (group(e) if kind == "group" else e["pos"]) == v
        per = [x for kind, v in keys for x in (sum(reward[b].sum() for b, e in enumerate(ents) if sel(kind, v, b, e)),
                                               G * sum(sel(kind, v, b, e) for b, e in enumerate(ents)))]
        r_ok = reward[ok]
        tot = torch.tensor([r_ok.sum(), r_ok.max(1).sum() if len(r_ok) else 0.0, (r_ok < 0.9).sum(), r_ok.size, len(live), int(ok.sum()),
                            sum(ended), len(ended), stats["pg"], stats["ce"], stats["clip"], stats["ratio"], stats["n"],
                            stats["tok"], stats["tok_n"], stats["tok_clip"], stats["tokens"], stats["tok_abs"]] + per,
                           device=dev, dtype=torch.float64)
        dist.all_reduce(tot)
        r_sum, best, der, n_r, n_live, n_win, n_end, n_calls, pg, ce, clip, ratio, n, tok, tok_n, tok_clip, tokens, tok_abs = tot.tolist()[:18]
        per = tot.tolist()[18:]
        n = max(n, 1)
        rec = {f"reward_{kind}/{v}": per[2 * i] / per[2 * i + 1] for i, (kind, v) in enumerate(keys) if per[2 * i + 1]}
        rec |= {"reward/mean": r_sum / n_r, "reward/best_of_group": best / n_win, "reward/derail": der / n_r,
                "groups_live": n_live / n_win, "ended": n_end / n_calls, "pg": pg / n, "ce": ce / (args.sft_per_step * world),
                "clip_frac": clip / n, "ratio_dev": ratio / n, "grad_norm": float(gnorm), "lr": lr,
                "tok/loss": tok / n, "tok/credited_frac": tok_n / max(tokens, 1), "tok/clip_frac": tok_clip / max(tok_n, 1),
                "tok/mean_abs_credit": tok_abs / max(tok_n, 1),
                "s_rollout": t_roll, "s_reward": t_rew, "s_per_step": time.time() - t0 - paused, "paused_s": paused,
                "gpu/temp": torch.cuda.temperature(dev)}
        if step % args.eval_every == 0 or step == args.steps:
            rec |= evaluate()
            save()
        if rank == 0:
            wb.log(rec, step=step)
            if step % 10 == 0 or step <= 10:
                print(f"step {step} reward {rec['reward/mean']:.4f} live {rec['groups_live']:.2f} pg {rec['pg']:.4f} "
                      f"ce {rec['ce']:.4f} clip {rec['clip_frac']:.3f} ratio {rec['ratio_dev']:.5f} "
                      f"tok {rec['tok/loss']:.5f} credited {rec['tok/credited_frac']:.4f} |c| {rec['tok/mean_abs_credit']:.3f} tok_clip {rec['tok/clip_frac']:.3f} "
                      f"rollout {t_roll:.1f} s reward {t_rew:.1f} s {rec['s_per_step']:.1f} s/step", flush=True)
    rpool.close()
    if rank == 0:
        wb.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train"])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--init", help="train: SFT checkpoint (snap_*.pt or last.pt)")
    ap.add_argument("--out", required=True)
    bd = os.path.expanduser("~/Work/BasecallerData")
    # human : bacteria = 4 : 1 (shares of the reads), split within each as runs/k4.sh (part2 : v3h, Sowpati : Hall)
    ap.add_argument("--sources", nargs="+", default=[f"{bd}/viktor_win12288_part2:part2:0.172", "/data/viktor_win_v3h:part1:0.628",
                                                     f"{bd}/viktor_win12288_bact:bact:0.053", f"{bd}/viktor_win12288_hall:hall:0.147"],
                    help="prep: window index sources DIR:PART:WEIGHT")
    ap.add_argument("--exclude-ids", nargs="*", default=["runs/readeval/ids.txt", "runs/readeval_bact/ids.txt"],
                    help="prep: read id files never used (the read-level test sets)")
    ap.add_argument("--reads", type=int, default=40_000, help="prep: training entries (one chunk per read)")
    ap.add_argument("--eval-reads", type=int, default=1024, help="prep: evaluation entries (val split)")
    ap.add_argument("--sft-windows", type=int, default=16_000, help="prep: SFT replay windows")
    ap.add_argument("--edge-share", type=float, default=0.3, help="prep: share of first / last chunks")
    ap.add_argument("--edge-slack", type=int, default=2500, help="prep: first / last chunk only if the segment reaches "
                                                                 "within this many samples of the read end (bounds the free own bases there)")
    ap.add_argument("--pad", type=int, default=6000, help="prep: truth = bases within chunk k +- this many samples "
                                                          "(> --margin bases of signal)")
    ap.add_argument("--ref-margin", type=int, default=50, help="prep: ... + this many reference bases on both sides")
    ap.add_argument("--min-bases", type=int, default=200, help="prep: skip entries whose chunk k holds fewer labelled bases")
    ap.add_argument("--procs", type=int, default=16, help="prep: loader processes")
    ap.add_argument("--pool", help="train: directory holding readpool/ (default --out)")
    ap.add_argument("--eval", nargs="+", default=[f"{bd}/viktor_win12288_bact:bact:val", f"{bd}/viktor_win12288_hall:hall:val"],
                    help="train: window evaluation sources DIR:PART:SPLIT")
    ap.add_argument("--eval-windows", default=f"{bd}/viktor_v3_heldout_n2048.parquet", help="train: stratified evaluation set")
    ap.add_argument("--eval-ref", default=f"{bd}/reference/hg002v1.2.fa", help="train: reference of --eval-windows")
    ap.add_argument("--margin", type=int, default=400, help="neighbour bases beyond each cut in the scored sequence "
                                                            "(long enough that a gap in chunk k scores as deletions)")
    ap.add_argument("--reward-procs", type=int, default=8, help="train: reward processes per GPU")
    ap.add_argument("--homopolymer-cap", type=int, default=60, help="as basecall_viktor (0: off)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--reads-per-step", type=int, default=64, help="entries per GPU and step")
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--mix", default="low:0.30,mid:0.45,perfect:0.10,bact:0.15",
                    help="train: sampling share per class (human entries by greedy reward of --init: low < --low <= mid "
                         "< --perfect <= perfect; bact: all bacterial entries)")
    ap.add_argument("--low", type=float, default=0.97)
    ap.add_argument("--perfect", type=float, default=0.995)
    ap.add_argument("--mine-batch", type=int, default=768, help="train: decode batch of the mining pass")
    ap.add_argument("--mine-init", help="train: reuse the mining classes of this checkpoint (default --init)")
    ap.add_argument("--lr", type=float, default=3e-6, help="peak learning rate")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--decay-start", type=int, default=1400, help="1 - sqrt decay to 0 from this step to --steps")
    ap.add_argument("--eps", type=float, default=4e-4, help="GSPO sequence-ratio clip")
    ap.add_argument("--std-floor", type=float, default=0.003, help="advantage: group reward std floor")
    ap.add_argument("--tok-weight", type=float, default=0.0, help="token credit term weight (0: sequence-level only)")
    ap.add_argument("--tok-eps", type=float, default=0.2, help="token credit term: token-ratio clip")
    ap.add_argument("--tok-window", type=int, default=2, help="token credit: a candidate is right at a truth position "
                                                                "when it has no error within this many bases")
    ap.add_argument("--updates", type=int, default=2, help="optimiser steps per rollout")
    ap.add_argument("--sft-weight", type=float, default=0.05)
    ap.add_argument("--sft-per-step", type=int, default=32, help="SFT replay windows per GPU and step")
    ap.add_argument("--micro", type=int, default=32, help="sequences per forward/backward")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=1024)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--temp-pause", type=float, default=78)
    ap.add_argument("--temp-resume", type=float, default=72)
    ap.add_argument("--wandb-project", default="viktor")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.cmd == "prep":
        prep(args)
        return
    assert args.init, "train needs --init"
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(dev)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    train(args, dev, rank, world)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

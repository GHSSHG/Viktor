"""Labels of the windows that start before the aligned segment (index rows with hard & 4, make_windows with
VIKTOR_KEEP_HEAD=1): the read start, i.e. the ligation adapter.

The reference only labels the part from the segment start on. For the part before it, Dorado SUP decodes the
window (koi encoder + CRF head + koi beam search) and every called base is placed at its CRF frame. The called bases
before the segment start become the head of the label; inside them the verified adapter (ADAPTER, consensus of
untrimmed SUP calls) replaces the matching stretch (edlib infix alignment, <= MAX_ED edits). Only the adapter bases carry
weight 1; the other called bases (T tail before it, bases between adapter and segment) have weight 0, so they are context only (the adapter ends where the genome starts, so its last bases are not masked). Without a clean adapter match the whole head has
weight 0. Result columns of the index: label (head + reference), lw (weights, read weight not applied), lc (encoder
frame of every base, as base_centers). The index file is rewritten in place.

Example:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.make_head_labels --index DIR --model-dir MODEL [--limit-files 2]
"""

import argparse
import glob
import json
import os
import tomllib

import edlib
import numpy as np
import polars as pl
import pod5
import pysam
import torch

from scripts.basecall_crf import CRFHead, beam_search
from viktor import load_dorado_encoder
from viktor.data import STRIDE, PARTS, base_centers, base_weights, label, part_paths, pod5_path, read_signals
from viktor.chunking import DNA_TRIM_START

ADAPTER = "CCTGTACTTCGTTCAGTTACGTATTGCT"  # consensus of 1437 untrimmed SUP read starts, 100% agreement over the core
MAX_ED = 5
CRF_FRAME = 6


def adapter_map(pre):
    """Align ADAPTER to the called head. Returns (start, end, ADAPTER base -> head position or None) or None."""
    r = edlib.align(ADAPTER, pre, mode="HW", task="path", k=MAX_ED)
    if r["editDistance"] < 0:
        return None
    s, e = r["locations"][0]
    qi, ti, m = 0, s, []
    n = ""
    for ch in r["cigar"]:
        if ch.isdigit():
            n += ch
            continue
        for _ in range(int(n)):
            if ch in "=X":
                m.append(ti); qi += 1; ti += 1
            elif ch == "I":      # adapter base missing from the call
                m.append(None); qi += 1
            else:                # D: called base not in the adapter
                ti += 1
        n = ""
    return s, e, m


def head_label(call, frames, sstart):
    """call: called bases of the window, frames: CRF frame of each (window-local), sstart: segment start sample in
    the window. Returns (head bases, weights, encoder-frame centres, adapter found)."""
    t = CRF_FRAME * np.asarray(frames, np.float64) + CRF_FRAME / 2  # called base midpoint, window-local sample
    n_pre = int((t < sstart).sum())
    pre, tc = call[:n_pre], (t[:n_pre] - STRIDE / 2) / STRIDE
    w = np.zeros(n_pre, np.float32)
    am = adapter_map(pre) if n_pre >= len(ADAPTER) - MAX_ED else None
    if am is None:
        return pre, w, tc.astype(np.float32), False
    s, e, m = am
    cen = np.array([tc[j] if j is not None else np.nan for j in m])
    ok = ~np.isnan(cen)
    if ok.sum() < len(ADAPTER) // 2:
        return pre, w, tc.astype(np.float32), False
    cen = np.interp(np.arange(len(m)), np.flatnonzero(ok), cen[ok])
    out = pre[:s] + ADAPTER + pre[e + 1:]
    wo = np.concatenate([np.zeros(s), np.ones(len(ADAPTER)), np.zeros(n_pre - e - 1)]).astype(np.float32)
    co = np.concatenate([tc[:s], cen, tc[e + 1:]]).astype(np.float32)
    return out, wo, co, True


def process(path, args, enc, head, q, fa, dev):
    idx = pl.read_parquet(path).with_row_index("_i")
    cand = idx.filter((pl.col("hard") & 4) > 0)
    stats = {"file": os.path.basename(path), "head_windows": cand.height, "adapter": 0, "dropped": 0}
    if cand.height == 0:
        return stats
    dwell = np.load(os.path.join(os.path.dirname(path), "dwell", os.path.basename(path)[:-len(".parquet")] + ".npy"),
                    mmap_mode="r")
    out, drop = {}, set()
    rows = cand.to_dicts()
    with pod5.Reader(pod5_path(rows[0]["pod5"])) as reader:
        for i in range(0, len(rows), args.batch):
            g = rows[i:i + args.batch]
            xs = read_signals(reader, g, enc.mean, enc.stdev)
            x = torch.from_numpy(np.stack(xs)).to(dev)[:, None]
            with torch.no_grad():
                calls = beam_search(head(enc(x)), q["scale"], q["bias"])
            for r, (seq, _, moves) in zip(g, calls):
                try:
                    frames = np.flatnonzero(moves)
                    sstart = CRF_FRAME * r["f_first"] - (r["win"] + DNA_TRIM_START)
                    hb, hw, hc, found = head_label(seq, frames, sstart)
                    ref = label(fa, {**r, "label": None})
                    rw = base_weights({**r, "lw": None, "w": 1.0}, ref)
                    rw[0] = 1.0  # the segment start is no window edge here
                    rc = base_centers({**r, "lc": None}, dwell)
                    assert len(ref) == len(rc) == len(rw)
                except (AssertionError, KeyError, IndexError):
                    drop.add(r["_i"])
                    continue
                out[r["_i"]] = (hb + ref, np.concatenate([hw, rw]).tolist(), np.concatenate([hc, rc]).tolist())
                stats["adapter"] += found
    n = idx.height
    lab, lw, lc = [None] * n, [None] * n, [None] * n
    for k, (a, b, c) in out.items():
        lab[k], lw[k], lc[k] = a, b, c
    idx = (idx.with_columns(pl.Series("label", lab, pl.Utf8), pl.Series("lw", lw, pl.List(pl.Float32)),
                            pl.Series("lc", lc, pl.List(pl.Float32)))
           .filter(~pl.col("_i").is_in(list(drop))).drop("_i"))
    tmp = path + ".tmp"
    idx.write_parquet(tmp)
    os.replace(tmp, path)
    stats["dropped"] = len(drop)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--part", default="part1", choices=sorted(PARTS))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit-files", type=int, default=None)
    args = ap.parse_args()
    dev = torch.device("cuda")
    enc = load_dorado_encoder(args.model_dir).to(dev, torch.float16).eval().use_koi()
    head = CRFHead(args.model_dir).to(dev).eval()
    with open(os.path.join(args.model_dir, "config.toml"), "rb") as f:
        q = tomllib.load(f)["qscore"]
    fa = pysam.FastaFile(part_paths(args.part)[0])
    for path in sorted(glob.glob(os.path.join(args.index, "*.pod5.parquet")))[:args.limit_files]:
        print(json.dumps(process(path, args, enc, head, q, fa, dev)), flush=True)


if __name__ == "__main__":
    main()

"""Whole-read basecalling with a Viktor checkpoint, assembled the way Dorado SUP does it.

Chunking is Dorado's variable-chunk-size path for transformer models (12288 samples, overlap 600, remainder as a short
chunk, repeat-padded to 768 samples, padded frames dropped before decoding). Dorado stitches with the CRF moves; the
decoder has none, so each overlap is resolved by aligning the end of chunk i with the start of chunk i+1 and cutting both
at the middle of the aligned overlap (the same midpoint rule as dorado utils::stitch_chunks).

Example:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.basecall_viktor --model-dir MODEL --pod5 R.pod5 --read-ids ids.txt \
      --out calls.fa runs/gspo3/snap_1200.pt
"""

import argparse
import time

import edlib
import numpy as np
import pod5
import torch

from scripts.basecall_crf import STRIDE
from viktor import Basecaller, DecodeEngine, get_tokenizer, load_dorado_encoder
from viktor.chunking import chunk_read, repeat_pad, scale_record


def overlap_cut(a, b, ov_frames, rate):
    """a, b: calls of adjacent chunks; ov_frames: encoder frames shared by them; rate: bases per frame.
    Returns (cut_a, cut_b): keep a[:cut_a] + b[cut_b:]."""
    est = max(int(ov_frames * rate), 4)
    n_tail = min(len(a), int(est * 1.6) + 16)
    m = min(len(b), max(int(est * 0.5), 8))
    ok = n_tail >= m >= 8
    if ok:
        r = edlib.align(b[:m], a[len(a) - n_tail:], mode="HW", task="locations", k=m // 3)
        ok = r["editDistance"] >= 0
    if ok:
        p = len(a) - n_tail + r["locations"][0][0]          # a[p] aligns with b[0]
        q = a[p:]
        r = edlib.align(q, b[:int(len(q) * 1.4) + 16], mode="SHW", task="path", k=max(len(q) // 3, 4))
        ok = r["editDistance"] >= 0
    if not ok:  # no clean overlap: cut both at the expected middle
        return len(a) - est // 2, min(est - est // 2, len(b))
    half, ia, ib = len(q) // 2, 0, 0
    for n, op in _cigar(r["cigar"]):
        for _ in range(n):
            if ia >= half and op in "=X":
                return p + ia, ib
            ia += op in "=XI"  # edlib: I = extra base in the query (q, from a), D = extra base in the target (b)
            ib += op in "=XD"
    return p + half, ib


def _cigar(c):
    n = ""
    for ch in c:
        if ch.isdigit():
            n += ch
        else:
            yield int(n), ch
            n = ""


def stitch(calls, spans, n_samples, overlap):
    """calls: per-chunk sequences, spans: (start, end) samples of each chunk."""
    out, cut_a_prev = [], 0
    for i, s in enumerate(calls):
        if i + 1 < len(calls):
            ov = (spans[i][1] - spans[i + 1][0]) // STRIDE
            rate = len(s) / max((spans[i][1] - spans[i][0]) // STRIDE, 1)
            ca, cb = overlap_cut(s, calls[i + 1], ov, rate)
            out.append(s[cut_a_prev:ca])
            cut_a_prev = cb
        else:
            out.append(s[cut_a_prev:])
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snap", help="snap_*.pt / last.pt checkpoint")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--pod5", required=True)
    ap.add_argument("--read-ids", help="file with one read id per line (default: all reads)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--overlap", type=int, help="chunk overlap in samples (default: the model's, 600)")
    ap.add_argument("--anchor-tail", action=argparse.BooleanOptionalAction, default=True,
                    help="reads >= one chunk: replace the short last chunk by a full chunk ending at the read end "
                         "(the decoder was only trained on full-length windows)")
    ap.add_argument("--dump-chunks", help="also write every chunk call (before stitching) to this FASTA")
    args = ap.parse_args()

    dev = torch.device("cuda")
    enc = load_dorado_encoder(args.model_dir).to(dev, torch.float16).eval().use_koi()
    s = torch.load(args.snap, map_location="cpu", weights_only=False)
    model = Basecaller(enc, get_tokenizer(s["args"].get("tokenizer", "base")))
    model.projector.load_state_dict(s["projector"])
    model.decoder.load_state_dict(s["decoder"])
    model.projector.to(dev, torch.bfloat16)
    model.decoder.to(dev, torch.bfloat16)
    model.eval()
    max_frames = enc.chunksize // enc.stride
    engine = DecodeEngine(model, args.batch, 2 * max_frames, max_frames)

    overlap = args.overlap or enc.overlap
    ids = [l.strip() for l in open(args.read_ids)] if args.read_ids else None
    reads, items = {}, []  # items: (read_id, chunk_idx, start, end, padded signal)
    with pod5.Reader(args.pod5) as r:
        for rec in r.reads(selection=ids):
            x = scale_record(rec, enc.mean, enc.stdev)
            rid = str(rec.read_id)
            spans = chunk_read(len(x), enc.chunksize, overlap)
            if args.anchor_tail and len(x) >= enc.chunksize and spans[-1][1] - spans[-1][0] < enc.chunksize:
                spans[-1] = (len(x) - enc.chunksize, len(x))
            reads[rid] = (len(x), spans)
            for ci, (a, b) in enumerate(spans):
                items.append((rid, ci, a, b, repeat_pad(x[a:b], enc.granularity)))
    items.sort(key=lambda it: -len(it[4]))
    print(f"{len(reads)} reads, {len(items)} chunks", flush=True)

    calls, t0 = {}, time.time()
    with torch.no_grad():
        for i in range(0, len(items), args.batch):
            g = items[i:i + args.batch]
            n = len(g)
            g = g + [g[-1]] * (args.batch - n)
            sig = torch.from_numpy(np.concatenate([it[4] for it in g])).to(dev, torch.float16)
            frames, flens = enc.encode_chunks(sig, [len(it[4]) for it in g], [it[3] - it[2] for it in g])
            toks = engine.generate(frames, flens).cpu()
            for it, t in zip(g[:n], toks):
                calls[(it[0], it[1])] = model.tokenizer.decode(t)
    print(f"decoded in {time.time() - t0:.0f} s", flush=True)

    if args.dump_chunks:
        with open(args.dump_chunks, "w") as f:
            for (rid, ci), seq in calls.items():
                a, b = reads[rid][1][ci]
                f.write(f">{rid}_{ci}_{a}_{b}_{reads[rid][0]}\n{seq}\n")
    with open(args.out, "w") as f:
        for rid, (n, spans) in reads.items():
            seq = stitch([calls[(rid, c)] for c in range(len(spans))], spans, n, overlap)
            if seq:
                f.write(f">{rid}\n{seq}\n")


if __name__ == "__main__":
    main()

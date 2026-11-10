"""Whole-read basecalling with a Viktor checkpoint, assembled the way Dorado SUP does it.

Chunking is Dorado's variable-chunk-size path for transformer models (12288 samples, overlap 600, remainder as a short
chunk, repeat-padded to 768 samples, padded frames dropped before decoding). Dorado stitches with the CRF moves; the
decoder has none, so each overlap is resolved by aligning the end of chunk i with the start of chunk i+1 and cutting both
at the middle of the aligned overlap (the same midpoint rule as dorado utils::stitch_chunks; overlap_cut has the
fallbacks).

Example:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.basecall_viktor --model-dir MODEL --pod5 R.pod5 --read-ids ids.txt \
      --out calls.fa runs/gspo3/snap_1200.pt
"""

import argparse
import re
import time

import edlib
import numpy as np
import pod5
import torch

from scripts.basecall_crf import STRIDE
from viktor import Basecaller, DecodeEngine, load_dorado_encoder
from scripts.make_head_labels import ADAPTER
from viktor.chunking import chunk_read, repeat_pad, scale_record
from viktor.fast_encoder import FastEncoder, FastEncoderConfig


def overlap_cut(a, b, ov_frames, rate):
    """a, b: calls of adjacent chunks; ov_frames: encoder frames shared by them; rate: bases per frame.
    Returns (cut_a, cut_b): keep a[:cut_a] + b[cut_b:]. The overlap is found by locating the head of b in the tail of a;
    when that fails (the real overlap is shorter than expected: a chunk end called short), by the best suffix(a) ~
    prefix(b) overlap. An (almost) empty chunk or no overlap at all: both kept whole (cutting at the expected middle
    dropped ~25 bases at such junctions)."""
    if len(a) < 8 or len(b) < 8:
        return len(a), 0
    est = max(int(ov_frames * rate), 4)
    n_tail = min(len(a), int(est * 1.6) + 16)
    m = min(len(b), max(int(est * 0.5), 8))
    if n_tail >= m:
        r = edlib.align(b[:m], a[len(a) - n_tail:], mode="HW", task="locations", k=m // 3)
        if r["editDistance"] >= 0:
            cut = _mid_cut(a, b, len(a) - n_tail + r["locations"][0][0])  # a[p] aligns with b[0]
            if cut:
                return cut
    best = None  # suffix(a) ~ prefix(b), at most 1 edit per 5 bases, score L - 4 x edits
    for L in range(min(n_tail, len(b)), 9, -1):
        ed = edlib.align(a[-L:], b[:L], mode="NW")["editDistance"]
        if ed <= L // 5 and (best is None or L - 4 * ed > best[0]):
            best = (L - 4 * ed, L)
    cut = best and _mid_cut(a, b, len(a) - best[1])
    return cut or (len(a), 0)


def _mid_cut(a, b, p):
    """a[p:] overlaps the head of b: cut both at the middle of their alignment (None: they do not align)."""
    q = a[p:]
    r = edlib.align(q, b[:int(len(q) * 1.4) + 16], mode="SHW", task="path", k=max(len(q) // 3, 4))
    if r["editDistance"] < 0:
        return None
    half, ia, ib = len(q) // 2, 0, 0
    for n, op in _cigar(r["cigar"]):
        for _ in range(n):
            if ia >= half and op in "=X":
                return p + ia, ib
            ia += op in "=XI"  # edlib: I = extra base in the query (q, from a), D = extra base in the target (b)
            ib += op in "=XD"
    return p + half, ib


def trim_adapter(seq, max_ed=8, head=120):
    """Cut the ligation adapter (and anything before it) off the read start, as dorado basecaller does by default; the
    model is trained to call it (make_head_labels.ADAPTER). No clean match: the read is left as called."""
    r = edlib.align(ADAPTER, seq[:head], mode="HW", task="locations", k=max_ed)
    return seq[r["locations"][0][1] + 1:] if r["editDistance"] >= 0 else seq


def cap_homopolymers(seq, max_run, keep=20):
    """A decoder that never emits EOS on a flat signal fills the chunk with one base (up to the generation cap); shorten
    every run of >= max_run identical bases to keep. Runs this long are not real DNA (and they are all there is to the junk)."""
    return re.sub(rf"(.)\1{{{max_run - 1},}}", lambda m: m.group(1) * keep, seq)


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
    ap.add_argument("--encoder", choices=["koi", "torch"], default="koi", help="encoder kernels (torch: the PyTorch layers, fp16)")
    ap.add_argument("--overlap", type=int, help="chunk overlap in samples (default: the model's, 600)")
    ap.add_argument("--anchor-tail", action=argparse.BooleanOptionalAction, default=True,
                    help="reads >= one chunk: replace the short last chunk by a full chunk ending at the read end "
                         "(the decoder was only trained on full-length windows)")
    ap.add_argument("--trim-adapter", action=argparse.BooleanOptionalAction, default=True,
                    help="cut the ligation adapter off the read start (dorado does by default)")
    ap.add_argument("--homopolymer-cap", type=int, default=60, metavar="N",
                    help="shorten runs of >= N identical bases in every chunk call to 20 before stitching (0: off)")
    ap.add_argument("--timing", help="write one TSV row per decoded batch: size, encode s, decode s, call lengths")
    ap.add_argument("--dump-chunks", help="also write every chunk call (before stitching) to this FASTA")
    args = ap.parse_args()

    dev = torch.device("cuda")
    s = torch.load(args.snap, map_location="cpu", weights_only=False)
    enc = load_dorado_encoder(args.model_dir)
    reference_stride = enc.stride
    fast = s["args"].get("encoder_arch") == "fast"
    if fast:
        enc = FastEncoder(FastEncoderConfig(**s["encoder_cfg"]), mean=enc.mean, stdev=enc.stdev)
    if "encoder" in s:  # fine-tuned (train.py --train-encoder)
        enc.load_state_dict(s["encoder"])
    enc = enc.to(dev, torch.float16).eval()
    if args.encoder == "koi" and not fast:
        enc.use_koi()
    model = Basecaller.from_args(enc, s["args"])
    model.projector.load_state_dict(s["projector"])
    model.decoder.load_state_dict(s["decoder"])
    model.projector.to(dev, torch.bfloat16)
    model.decoder.to(dev, torch.bfloat16)
    model.eval()
    max_frames = enc.chunksize // enc.stride
    # Keep the original generation budget in signal units. The FAST prefix is
    # four times shorter, but the number of bases to generate has not changed.
    engine = DecodeEngine(model, args.batch, 2 * (enc.chunksize // reference_stride), max_frames)

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

    calls, t0, tim = {}, time.time(), []
    with torch.no_grad():
        for i in range(0, len(items), args.batch):
            g = items[i:i + args.batch]
            n = len(g)
            g = g + [g[-1]] * (args.batch - n)
            t1 = time.time()
            if fast:
                size = max(len(it[4]) for it in g)
                signal = np.zeros((len(g), 1, size), dtype=np.float16)
                for b, it in enumerate(g):
                    signal[b, 0, :len(it[4])] = it[4]
                sig = torch.from_numpy(signal).to(dev)
                frames, flens = model.encode(sig, torch.tensor([it[3] - it[2] for it in g], device=dev))
            else:
                sig = torch.from_numpy(np.concatenate([it[4] for it in g])).to(dev, torch.float16)
                frames, flens = enc.encode_chunks(sig, [len(it[4]) for it in g], [it[3] - it[2] for it in g])
            torch.cuda.synchronize()
            t2 = time.time()
            toks = engine.generate(frames, flens).cpu()
            t3 = time.time()
            for it, t in zip(g[:n], toks):
                calls[(it[0], it[1])] = model.tokenizer.decode(t)
                if args.homopolymer_cap:
                    calls[(it[0], it[1])] = cap_homopolymers(calls[(it[0], it[1])], args.homopolymer_cap)
            tim.append((n, t2 - t1, t3 - t2, [len(calls[(it[0], it[1])]) for it in g[:n]]))
    print(f"decoded in {time.time() - t0:.0f} s", flush=True)
    if args.timing:
        with open(args.timing, "w") as f:
            for n, te, td, lens in tim:
                f.write(f"{n}\t{te:.4f}\t{td:.4f}\t{','.join(map(str, lens))}\n")

    if args.dump_chunks:
        with open(args.dump_chunks, "w") as f:
            for (rid, ci), seq in calls.items():
                a, b = reads[rid][1][ci]
                f.write(f">{rid}_{ci}_{a}_{b}_{reads[rid][0]}\n{seq}\n")
    with open(args.out, "w") as f:
        for rid, (n, spans) in reads.items():
            seq = stitch([calls[(rid, c)] for c in range(len(spans))], spans, n, overlap)
            if args.trim_adapter:
                seq = trim_adapter(seq)
            if seq:
                f.write(f">{rid}\n{seq}\n")


if __name__ == "__main__":
    main()

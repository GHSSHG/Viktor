"""Basecall a pod5 with the Dorado SUP encoder (PyTorch or koi) + Dorado's upsample/CRF head in PyTorch.

Chunking and stitching follow Dorado's variable-chunk-size path for transformer models
(utils::generate_variable_chunks_tx, BasecallerNode repeat-padding to 768 samples, stitch_chunks).
Decoding: Dorado's own GPU beam search (koi, CUDADecoder defaults: beam width 32, beam cut 100,
blank 2.0), or a free Viterbi over the CRF (layout as SUPEncoder/crf.py).

Example:
  CUDA_VISIBLE_DEVICES=1 python scripts/basecall_crf.py --model-dir MODEL --pod5 R.pod5 --encoder torch --out calls.fa
"""

import argparse
import os
import sys

import numpy as np
import pod5
import torch
import torch.nn.functional as F

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from viktor import load_dorado_encoder
from viktor.encoder import _load_tensor

STRIDE, NB, NS = 6, 4, 4 ** 5
GRANULARITY = 768  # stride_inner (12) * 64
BATCH = 256


def chunk_read(n, size, overlap):
    """generate_variable_chunks_tx: full chunks every size - overlap, then the remainder."""
    out, off = [], 0
    while off + size < n:
        out.append((off, off + size))
        off += size - overlap
    if off != n:
        out.append((off, n))
    return out


def repeat_pad(x, unit):
    while len(x) % unit:
        x = np.concatenate([x, x[:unit - len(x) % unit]])
    return x


class CRFHead(torch.nn.Module):
    """Dorado LinearUpsample + LinearScaledCRF in fp16 (weight pre-multiplied by scale in fp16)."""

    def __init__(self, model_dir, scale=5.0):
        super().__init__()
        t = lambda n: _load_tensor(os.path.join(model_dir, n + ".tensor")).half()
        self.register_buffer("up_w", t("upsample.linear.weight"))
        self.register_buffer("up_b", t("upsample.linear.bias"))
        self.register_buffer("crf_w", t("crf.linear.weight") * scale)

    def forward(self, h):  # h [N, T, 512] -> scores [N, 2T, 4096] fp16, blank score is the constant 2.0
        n, t, c = h.shape
        x = F.linear(h.half(), self.up_w, self.up_b).reshape(n, 2 * t, c)
        return F.linear(x, self.crf_w)


@torch.no_grad()
def viterbi(scores):
    """scores [N, F, 4096] -> per chunk (sequence, moves[F])."""
    n, nf = scores.shape[:2]
    scores = scores.view(n, nf, NS, NB)
    dev = scores.device
    j = torch.arange(NS, device=dev)
    pred = torch.stack([k * (NS // NB) + j // NB for k in range(NB)], 1)
    v = torch.zeros(n, NS, device=dev)
    bp = torch.empty(n, nf, NS, dtype=torch.uint8, device=dev)
    for t in range(nf):
        cand = torch.cat([(v + 2.0)[:, :, None], v[:, pred] + scores[:, t].float()], 2)
        v, bp[:, t] = cand.max(2)
    bp, predc, s_end = bp.cpu().numpy(), pred.cpu().numpy(), v.argmax(1).cpu().numpy()
    out = []
    for i in range(n):
        s, seq, moves = int(s_end[i]), [], np.zeros(nf, dtype=np.uint8)
        for t in range(nf - 1, -1, -1):
            a = bp[i, t, s]
            if a:
                seq.append("ACGT"[s % NB])
                moves[t] = 1
                s = predc[s, a - 1]
        out.append(("".join(reversed(seq)), moves))
    return out


def beam_search(scores, q_scale, q_shift):
    """Dorado CUDADecoder::beam_search_part_1/2 on koi. scores [N, F, 4096] -> per chunk (sequence, moves[F])."""
    from viktor.koi import load_extension
    msq = load_extension().beam_search(scores, 0.0, 32, 100.0, 2.0, q_scale, q_shift, False).cpu().numpy()
    out = []
    for i in range(scores.shape[0]):
        moves = msq[0, i].astype(np.uint8)
        out.append((msq[1, i, :int(moves.sum())].tobytes().decode(), moves))
    return out


def stitch(chunks, n_samples, overlap):
    """dorado utils::stitch_chunks with is_vcs=True. chunks: [(start, raw_size, seq, moves)]."""
    seqs, start_pos, mid_front = [], 0, 0
    for (s0, r0, seq0, mv0), (s1, _, _, mv1) in zip(chunks, chunks[1:]):
        ov = (r0 + s0 - s1) // STRIDE
        mid_rear = ov // 2
        end = len(seq0) - int(mv0[len(mv0) - mid_rear:].sum())
        seqs.append(seq0[start_pos:end])
        mid_front = ov - mid_rear
        start_pos = int(mv1[:mid_front].sum())
    _, r_last, seq_last, mv_last = chunks[-1]
    keep = -(-min(n_samples, r_last) // STRIDE)
    seqs.append(seq_last[start_pos:start_pos + int(mv_last[mid_front:keep].sum())])
    return "".join(seqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--pod5", required=True)
    ap.add_argument("--encoder", choices=["torch", "koi", "koi-f32acc"], required=True)
    ap.add_argument("--decoder", choices=["beam", "viterbi"], default="beam")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device("cuda")
    enc = load_dorado_encoder(args.model_dir).to(dev, torch.float16).eval()
    if args.encoder.startswith("koi"):
        enc.use_koi(swiglu_f32_accum=args.encoder == "koi-f32acc")
    head = CRFHead(args.model_dir).to(dev).eval()
    with open(os.path.join(args.model_dir, "config.toml"), "rb") as f:
        q = tomllib.load(f)["qscore"]
    decode = (lambda s: beam_search(s, q["scale"], q["bias"])) if args.decoder == "beam" else viterbi

    reads, items = {}, []  # items: (read_id, chunk_idx, start, raw_size, padded signal)
    with pod5.Reader(args.pod5) as r:
        for rec in r.reads():
            x = enc.standardise(rec.signal_pa.astype(np.float32))
            rid = str(rec.read_id)
            spans = chunk_read(len(x), enc.chunksize, enc.overlap)
            reads[rid] = (len(x), len(spans))
            for ci, (a, b) in enumerate(spans):
                items.append((rid, ci, a, b - a, repeat_pad(x[a:b], GRANULARITY)))

    results = {}
    by_len = {}
    for it in items:
        by_len.setdefault(len(it[4]), []).append(it)
    with torch.no_grad():
        for L, group in sorted(by_len.items()):
            for i in range(0, len(group), BATCH):
                g = group[i:i + BATCH]
                n_pad = -(-len(g) // 64) * 64  # koi batch granularity; extra rows are copies, discarded
                sig = np.stack([it[4] for it in g] + [g[0][4]] * (n_pad - len(g)))
                h = enc(torch.from_numpy(sig).to(dev, torch.float16)[:, None])
                for it, (seq, moves) in zip(g, decode(head(h)[:len(g)])):
                    results[(it[0], it[1])] = (it[2], it[3], seq, moves)

    with open(args.out, "w") as f:
        for rid, (n, n_chunks) in reads.items():
            seq = stitch([results[(rid, c)] for c in range(n_chunks)], n, enc.overlap)
            if seq:
                f.write(f">{rid}\n{seq}\n")
    print(f"{len(reads)} reads, {len(items)} chunks -> {args.out}")


if __name__ == "__main__":
    main()

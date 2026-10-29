"""Free-decoding evaluation of checkpoints and the Dorado SUP baseline on one fixed window set. Strict identity
1 - edit distance / max(call length, label length) (NW), over the whole window ('identity_*') and with TRIM label bases
dropped at each side that overlaps a neighbouring chunk ('trim_*': what Dorado's stitching keeps; a read start or end is
not trimmed; the call is cut where it aligns to the kept span); derailed = identity < 0.9. Windows: a make_windows
parquet (e.g. the external EF22 set) with the reference its labels come from. Calls are saved next to --out.
A parquet with a 'stratum' column (and 'pop', the stratum's window count in the population it was sampled from) is
scored per stratum, plus 'natural': the strata with labels averaged by pop. Windows labelled "" (no DNA) are scored
as the fraction called with at most EMPTY_MAX bases and the mean call length.

Example:
  CUDA_VISIBLE_DEVICES=3 python -m scripts.evaluate --model-dir MODEL --windows ~/Work/BasecallerData/viktor_ef22_n1024.parquet \
      --ref /data_nvme/bacteria_ef22/reference/ef22_ref.fasta --out runs/eval_ef22.json runs/sft2/snap_0020000.pt runs/sft2/snap_0030000.pt
"""

import argparse
import json

import edlib
import numpy as np
import polars as pl
import torch

from scripts.train import dorado_calls, heldout_calls
from viktor import Basecaller, get_tokenizer, load_dorado_encoder
from viktor.data import load_windows

EMPTY_MAX = 4
TRIM = 24  # label bases Dorado's stitching drops at each side of a 600-sample chunk overlap


def strict(pred, ref, row):
    """(identity, trimmed identity or None when no label base is left)."""
    lo = 0 if row.get("win") == 0 else TRIM
    hi = len(ref) if (row.get("hard") or 0) & 24 else len(ref) - TRIM
    cig = edlib.align(pred, ref, mode="NW", task="path")["cigar"] or f"{len(ref)}D"
    ri = ed = ed_mid = q_mid = 0
    num = ""
    for ch in cig:
        if ch.isdigit():
            num += ch
            continue
        n, num = int(num), ""
        mid = max(0, min(ri + n, hi) - max(ri, lo))  # reference bases of this run inside [lo, hi)
        if ch == "I":
            mid = n if lo <= ri < hi else 0
            ed, ed_mid, q_mid = ed + n, ed_mid + mid, q_mid + mid
            continue
        if ch != "=":
            ed, ed_mid = ed + n, ed_mid + mid
        if ch != "D":
            q_mid += mid
        ri += n
    return 1 - ed / max(len(pred), len(ref)), (1 - ed_mid / max(q_mid, hi - lo) if hi > lo else None)


def identity_stats(preds, wins, rows):
    ident, trim = zip(*(strict(p, s, r) for p, (_, s, _), r in zip(preds, wins, rows)))
    ident, trim = np.array(ident), np.array([t for t in trim if t is not None])
    st = lambda a: {"mean": float(a.mean()), "median": float(np.median(a)), "derail": float((a < 0.9).mean())}
    return ({f"identity_{k}": v for k, v in st(ident).items()} | {f"trim_{k}": v for k, v in st(trim).items()}
            | {"n_trim": len(trim)})


def scores(preds, wins, rows):
    if "stratum" not in rows[0]:
        return identity_stats(preds, wins, rows)
    out, nat = {}, {}
    for st in dict.fromkeys(r["stratum"] for r in rows):
        ids = [i for i, r in enumerate(rows) if r["stratum"] == st]
        if all(wins[i][1] == "" for i in ids):
            n = np.array([len(preds[i]) for i in ids])
            out[st] = {"n": len(ids), "empty_ok": float((n <= EMPTY_MAX).mean()), "call_len_mean": float(n.mean())}
        else:
            out[st] = {"n": len(ids)} | identity_stats(*([a[i] for i in ids] for a in (preds, wins, rows)))
            nat[st] = rows[ids[0]]["pop"]
    tot = sum(nat.values())
    out["natural"] = {k: sum(out[st][k] * n for st, n in nat.items()) / tot for k in ("identity_mean", "identity_derail", "trim_mean", "trim_derail")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snaps", nargs="*", help="snap_*.pt / last.pt checkpoints")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--windows", required=True, help="make_windows parquet")
    ap.add_argument("--ref", required=True, help="reference FASTA of the windows' labels")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", required=True, help="results JSON")
    args = ap.parse_args()

    dev = torch.device("cuda")
    encoder = load_dorado_encoder(args.model_dir).to(dev, torch.float16).eval().use_koi()
    rows = pl.read_parquet(args.windows).to_dicts()
    wins = load_windows(rows, encoder.mean, encoder.stdev, args.ref)
    calls = {"dorado": dorado_calls(args.model_dir, encoder, wins, dev)}
    res = {"windows": args.windows, "n": len(wins), "dorado": scores(calls["dorado"], wins, rows)}
    print("dorado", json.dumps(res["dorado"]), flush=True)
    for path in args.snaps:
        s = torch.load(path, map_location="cpu", weights_only=False)
        model = Basecaller(encoder, get_tokenizer(s["args"].get("tokenizer", "base")))
        model.projector.load_state_dict(s["projector"])
        model.decoder.load_state_dict(s["decoder"])
        model.projector.to(dev)
        model.decoder.to(dev)
        calls[path] = heldout_calls(model, wins, args.batch, dev)
        r = scores(calls[path], wins, rows)
        res[f"{path} (step {s['step']})"] = r
        print(path, s["step"], json.dumps(r), flush=True)
        del model
        torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    pl.DataFrame(calls).write_parquet(args.out.removesuffix(".json") + "_calls.parquet")


if __name__ == "__main__":
    main()

"""Free-decoding evaluation of checkpoints and the Dorado SUP baseline on one fixed window set, scored exactly as the
training heldout evaluation (scripts.train.identity_stats: global identity, identity over scored bases, derailed
fraction < 0.9). Windows: a make_windows parquet (e.g. the external EF22 set) with the reference its labels come from.
A parquet with a 'stratum' column (and 'pop', the stratum's window count in the population it was sampled from) is
scored per stratum, plus 'natural': the strata with labels averaged by pop. Windows labelled "" (no DNA) are scored
as the fraction called with at most EMPTY_MAX bases and the mean call length.

Example:
  CUDA_VISIBLE_DEVICES=3 python -m scripts.evaluate --model-dir MODEL --windows ~/Work/BasecallerData/viktor_ef22_n1024.parquet \
      --ref /data_nvme/bacteria_ef22/reference/ef22_ref.fasta --out runs/eval_ef22.json runs/sft2/snap_0020000.pt runs/sft2/snap_0030000.pt
"""

import argparse
import json

import numpy as np
import polars as pl
import torch

from scripts.train import dorado_calls, heldout_calls, identity_stats
from viktor import Basecaller, get_tokenizer, load_dorado_encoder
from viktor.data import load_windows

EMPTY_MAX = 4


def scores(preds, wins, rows):
    if "stratum" not in rows[0]:
        return identity_stats(preds, wins)
    out, nat = {}, {}
    for st in dict.fromkeys(r["stratum"] for r in rows):
        ids = [i for i, r in enumerate(rows) if r["stratum"] == st]
        if all(wins[i][1] == "" for i in ids):
            n = np.array([len(preds[i]) for i in ids])
            out[st] = {"n": len(ids), "empty_ok": float((n <= EMPTY_MAX).mean()), "call_len_mean": float(n.mean())}
        else:
            out[st] = {"n": len(ids)} | identity_stats([preds[i] for i in ids], [wins[i] for i in ids])
            nat[st] = rows[ids[0]]["pop"]
    tot = sum(nat.values())
    out["natural"] = {k: sum(out[st][k] * n for st, n in nat.items()) / tot for k in ("identity_mean", "derail", "masked_mean")}
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
    res = {"windows": args.windows, "n": len(wins), "dorado": scores(dorado_calls(args.model_dir, encoder, wins, dev), wins, rows)}
    print("dorado", json.dumps(res["dorado"]), flush=True)
    for path in args.snaps:
        s = torch.load(path, map_location="cpu", weights_only=False)
        model = Basecaller(encoder, get_tokenizer(s["args"].get("tokenizer", "base")))
        model.projector.load_state_dict(s["projector"])
        model.decoder.load_state_dict(s["decoder"])
        model.projector.to(dev)
        model.decoder.to(dev)
        r = scores(heldout_calls(model, wins, args.batch, dev), wins, rows)
        res[f"{path} (step {s['step']})"] = r
        print(path, s["step"], json.dumps(r), flush=True)
        del model
        torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

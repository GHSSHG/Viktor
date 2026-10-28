"""Free-decoding evaluation of checkpoints and the Dorado SUP baseline on one fixed window set, scored exactly as the
training heldout evaluation (scripts.train.identity_stats: global identity, identity over scored bases, derailed
fraction < 0.9). Windows: a make_windows parquet (e.g. the external EF22 set) with the reference its labels come from.

Example:
  CUDA_VISIBLE_DEVICES=3 python -m scripts.evaluate --model-dir MODEL --windows ~/Work/BasecallerData/viktor_ef22_n1024.parquet \
      --ref /data_nvme/bacteria_ef22/reference/ef22_ref.fasta --out runs/eval_ef22.json runs/sft2/snap_0020000.pt runs/sft2/snap_0030000.pt
"""

import argparse
import json

import polars as pl
import torch

from scripts.train import dorado_calls, eval_heldout, identity_stats
from viktor import Basecaller, get_tokenizer, load_dorado_encoder
from viktor.data import load_windows


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
    wins = load_windows(pl.read_parquet(args.windows).to_dicts(), encoder.mean, encoder.stdev, args.ref)
    res = {"windows": args.windows, "n": len(wins),
           "dorado": identity_stats(dorado_calls(args.model_dir, encoder, wins, dev), wins)}
    print("dorado", json.dumps(res["dorado"]), flush=True)
    for path in args.snaps:
        s = torch.load(path, map_location="cpu", weights_only=False)
        model = Basecaller(encoder, get_tokenizer(s["args"].get("tokenizer", "base")))
        model.projector.load_state_dict(s["projector"])
        model.decoder.load_state_dict(s["decoder"])
        model.projector.to(dev)
        model.decoder.to(dev)
        r = {k.split("/", 1)[1]: v for k, v in eval_heldout(model, wins, args.batch, dev).items()}
        res[f"{path} (step {s['step']})"] = r
        print(path, s["step"], json.dumps(r), flush=True)
        del model
        torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

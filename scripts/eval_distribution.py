"""Per-window accuracy distribution (not just median/mean) for Viktor checkpoints and the Dorado SUP baseline.
Saves per-window global identity and masked identity so the mode can be estimated. Small batches (shares a GPU with training).

  CUDA_VISIBLE_DEVICES=1 python -m scripts.eval_distribution --model-dir MODEL --set hg002 --out runs/dist_hg002.npz runs/sft2/snap_0210000.pt
  CUDA_VISIBLE_DEVICES=1 python -m scripts.eval_distribution --model-dir MODEL --set ef22 --out runs/dist_ef22.npz runs/sft2/snap_0190000.pt
"""
import argparse, copy, os, tomllib
import edlib, numpy as np, polars as pl, torch
from scripts.train import fixed_windows, masked_identity
from viktor import Basecaller, get_tokenizer, load_dorado_encoder
from viktor.data import collate, load_windows
from viktor.engine import DecodeEngine

def identities(preds, wins):
    g = np.array([1 - edlib.align(p, s, mode="NW", task="distance")["editDistance"] / len(s) for p, (_, s, _) in zip(preds, wins)])
    m = np.array([masked_identity(p, s, w > 0) for p, (_, s, w) in zip(preds, wins)])
    return g, m

@torch.no_grad()
def dorado_preds(model_dir, encoder, wins, dev, bs):
    from scripts.basecall_crf import CRFHead, beam_search
    head = CRFHead(model_dir).to(dev).eval()
    q = tomllib.load(open(os.path.join(model_dir, "config.toml"), "rb"))["qscore"]
    out = []
    for i in range(0, len(wins), bs):
        x = torch.from_numpy(np.stack([w[0] for w in wins[i:i + bs]])).to(dev)[:, None]
        out += [c[0] for c in beam_search(head(encoder(x)), q["scale"], q["bias"])]
    del head; torch.cuda.empty_cache()
    return out

@torch.no_grad()
def viktor_preds(model, wins, dev, bs):
    inf = Basecaller(model.encoder, model.tokenizer, model.cfg)
    inf.projector = copy.deepcopy(model.projector).to(torch.bfloat16).eval()
    inf.decoder = copy.deepcopy(model.decoder).to(torch.bfloat16).eval()
    eng = DecodeEngine(inf, bs, max(len(s) for _, s, _ in wins) + 64, model.encoder.chunksize // model.encoder.stride)
    preds = []
    for i in range(0, len(wins), bs):
        g = wins[i:i + bs]; g = g + [g[-1]] * (bs - len(g))
        fr = model.encoder(collate(g)[0].to(dev))
        toks = eng.generate(fr.reshape(-1, fr.shape[-1]), [fr.shape[1]] * bs).cpu()
        preds += [model.tokenizer.decode(t) for t in toks]
    del eng, inf; torch.cuda.empty_cache()
    return preds[:len(wins)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snaps", nargs="*"); ap.add_argument("--model-dir", required=True)
    ap.add_argument("--set", choices=["hg002", "ef22", "parquet"], required=True)
    ap.add_argument("--windows"); ap.add_argument("--ref"); ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--n", type=int, default=1024)
    a = ap.parse_args()
    dev = torch.device("cuda")
    enc = load_dorado_encoder(a.model_dir).to(dev, torch.float16).eval().use_koi()
    if a.set == "hg002":
        wins = fixed_windows(os.path.expanduser("~/Work/BasecallerData/viktor_win12288"), "heldout", a.n, enc.mean, enc.stdev)
    elif a.set == "parquet":
        wins = load_windows(pl.read_parquet(os.path.expanduser(a.windows)).to_dicts(), enc.mean, enc.stdev, a.ref)
    else:
        wins = load_windows(pl.read_parquet(os.path.expanduser("~/Work/BasecallerData/viktor_ef22_n1024.parquet")).to_dicts(),
                            enc.mean, enc.stdev, "/data_nvme/bacteria_ef22/reference/ef22_ref.fasta")
    res = {}
    g, m = identities(dorado_preds(a.model_dir, enc, wins, dev, a.batch), wins); res["dorado_global"], res["dorado_masked"] = g, m
    print("dorado median", np.median(m), "n", len(m), flush=True)
    for p in a.snaps:
        s = torch.load(p, map_location="cpu", weights_only=False)
        model = Basecaller(enc, get_tokenizer("base"))
        model.projector.load_state_dict(s["projector"]); model.decoder.load_state_dict(s["decoder"])
        model.projector.to(dev); model.decoder.to(dev)
        g, m = identities(viktor_preds(model, wins, dev, a.batch), wins)
        k = f"viktor_{s['step']}"; res[k + "_global"], res[k + "_masked"] = g, m
        print(k, "median", np.median(m), flush=True)
        del model; torch.cuda.empty_cache()
    np.savez(a.out, **res)

if __name__ == "__main__":
    main()

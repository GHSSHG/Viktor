import os
import random

import pytest
import torch

from viktor import (BOS, EOS, PAD, Basecaller, DecodeEngine, DecoderConfig, DoradoEncoder,
                    KmerTokenizer, load_dorado_encoder)

MODEL_DIR = os.environ.get("VIKTOR_DORADO_MODEL")

TINY_CFG = {
    "model": {"encoder": {
        "conv": {"sublayers": [
            {"type": "convolution", "insize": 1, "size": 16, "winlen": 5, "stride": 1, "padding": 2, "bias": True},
            {"type": "convolution", "insize": 16, "size": 32, "winlen": 9, "stride": 4, "padding": 4, "bias": True},
            {"type": "permute", "dims": [0, 2, 1]}]},
        "transformer_encoder": {"depth": 2, "layer": {
            "d_model": 32, "nhead": 4, "dim_feedforward": 64, "deepnorm_alpha": 2.0, "attn_window": [7, 8]}}}},
    "basecaller": {"chunksize": 512, "overlap": 16},
}


def tiny_model(k, seed=0):
    torch.manual_seed(seed)
    tok = KmerTokenizer(k)
    cfg = DecoderConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                        head_dim=8, ffn_dim=64, max_seq_len=512)
    return Basecaller(DoradoEncoder(TINY_CFG), tok, cfg).eval()


def greedy_reference(model, frames_list, max_new):
    """Recompute the whole sequence every step, no cache; one prefix per sequence."""
    eng_bias = torch.zeros(model.tokenizer.vocab_size)
    eng_bias[[PAD, BOS]] = float("-inf")
    out = []
    for frames in frames_list:
        prefix = model.projector(frames)[None]
        toks, prev = [], BOS
        for _ in range(max_new):
            if prev in (EOS, PAD):
                nxt = PAD
            elif model.tokenizer.tail_mask[prev]:
                nxt = EOS
            else:
                x = torch.cat([prefix, model.decoder.embed(torch.tensor([[BOS] + toks]))], 1)
                logits = model.decoder.head(model.decoder(x)[:, -1])[0]
                nxt = int((logits + eng_bias).argmax())
            if nxt not in (EOS, PAD):
                toks.append(nxt)
            out.append(nxt)
            prev = nxt
    return torch.tensor(out).view(len(frames_list), max_new)


@pytest.mark.parametrize("k", [1, 4])
def test_tokenizer_roundtrip(k):
    tok = KmerTokenizer(k)
    assert tok.vocab_size == 3 + sum(4 ** j for j in range(1, k + 1))
    rng = random.Random(k)
    for n in range(0, 23):
        s = "".join(rng.choice("ACGT") for _ in range(n))
        ids = tok.encode(s)
        assert len(ids) == -(-n // k)
        assert not any(tok.tail_mask[i] for i in ids[:-1])
        assert tok.decode(ids + [EOS, 5]) == s
        assert int(tok.n_bases[ids].sum()) == n


@pytest.mark.parametrize("k", [1, 4])
def test_engine_varlen_matches_full_recompute(k):
    model = tiny_model(k)
    lens = [256, 512, 256]  # multiples of the tiny granularity (stride 4 * 64)
    with torch.no_grad():
        sig = torch.randn(sum(lens))
        frames = model.encoder.forward_varlen(sig, lens)
        flens = [n // model.encoder.stride for n in lens]
        assert frames.shape[0] == sum(flens)
        eng = DecodeEngine(model, 3, 20, max(flens), backend="torch", prefill_batch=2)
        got = eng.generate(frames, flens).clone()
        ref = greedy_reference(model, list(frames.split(flens)), 20)
    assert torch.equal(got, ref)


def test_encoder_varlen_is_per_chunk():
    model = tiny_model(1)
    lens = [512, 256]
    with torch.no_grad():
        sig = torch.randn(sum(lens))
        packed = model.encoder.forward_varlen(sig, lens)
        alone = torch.cat([model.encoder(sig[:512].view(1, 1, -1))[0], model.encoder(sig[512:].view(1, 1, -1))[0]])
    assert torch.allclose(packed, alone)


def test_encode_chunks_drops_padding_frames():
    model = tiny_model(1)
    lens, valid = [512, 256], [512, 130]  # second chunk: 130 real samples, repeat-padded to 256
    with torch.no_grad():
        sig = torch.randn(sum(lens))
        packed = model.encoder.forward_varlen(sig, lens)
        frames, flens = model.encoder.encode_chunks(sig, lens, valid)
    assert flens == [128, 33]  # ceil(130 / stride 4)
    assert torch.equal(frames, torch.cat([packed[:128], packed[128:161]]))


def test_loss_backward_and_freeze():
    model = tiny_model(4).train()
    model.set_encoder_trainable(False)
    loss = model.loss(torch.randn(2, 1, 256), ["ACGTACGTA", "TTGCA"])
    loss.backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model.decoder.lm_head.weight.grad is not None


@pytest.mark.skipif(MODEL_DIR is None, reason="set VIKTOR_DORADO_MODEL to a Dorado SUP model dir")
def test_load_dorado_weights():
    enc = load_dorado_encoder(MODEL_DIR).eval()
    assert enc.stride == 12 and enc.d_model == 512
    with torch.no_grad():
        y = enc(torch.randn(1, 1, enc.chunksize))
    assert y.shape == (1, enc.chunksize // 12, 512) and torch.isfinite(y).all()

"""Static-interface GPU regression: real sliding FlashAttention vs masked SDPA,
variable lengths, padding isolation, frame centres, and equal decoder init.
Run on one GPU before the two-rank real-data trainer smoke.
"""
import copy
import json
import sys
import argparse
import torch
import numpy as np
from viktor.mature_encoder import BottleneckTransformer, MatureEncoder
from viktor import load_dorado_encoder
from scripts.train_mature import build_model, convert_centers


def main():
    torch.manual_seed(23)
    dev = torch.device('cuda')
    for width, window in [(1024,64),(2048,32)]:
        flash = BottleneckTransformer(width,window).to(dev)
        dense = copy.deepcopy(flash)
        # Force the reference branch without disabling FlashAttention elsewhere.
        import viktor.mature_encoder as mod
        original = mod.flash_attn_varlen_func
        x = torch.randn(3,129,width,device=dev,requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        lens = torch.tensor([129,73,0],device=dev)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            actual = flash(x,lens)
            mod.flash_attn_varlen_func = None
            dense.require_flash = False
            expected = dense(y,lens)
            mod.flash_attn_varlen_func = original
        torch.testing.assert_close(actual,expected,atol=.06,rtol=.04)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(x.grad,y.grad,atol=2e-5,rtol=.1)
        assert actual[1,73:].abs().max()==0 and actual[2].abs().max()==0
        modified = x.detach().clone()
        modified[1,73:] = 1000
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            isolated = flash(modified,lens)
        torch.testing.assert_close(actual[1,:73],isolated[1,:73],atol=1e-5,rtol=1e-5)
        print('PASS attention outputs/gradients/padding',width,flush=True)
        del flash,dense,x,y,actual,expected,modified,isolated
    args = argparse.Namespace(model_dir=sys.argv[1],encoder_init=sys.argv[2],compression=1,
                              seed=0,tokenizer='4mer',layers=4,prefix_pos_scale=1,align_sigma=4.)
    a = build_model(args,dev)
    fingerprint = {n:p.detach().cpu().clone() for n,p in a.decoder.named_parameters()}
    del a
    args.compression,args.prefix_pos_scale,args.align_sigma=4,4,1.
    b=build_model(args,dev)
    assert all(torch.equal(p.cpu(),fingerprint[n]) for n,p in b.decoder.named_parameters())
    sig=torch.randn(3,1,1536,device=dev)
    valid=torch.tensor([1536,801,500],device=dev)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        frames,lens=b.encode(sig,valid)
        single,slens=b.encode(sig[1:2],valid[1:2])
    assert lens.tolist()==[32,17,11]
    torch.testing.assert_close(frames[1,:17],single[0,:17],atol=.03,rtol=.03)
    assert frames[1,17:].abs().max()==0
    assert np.allclose(convert_centers([np.array([-1,0,1.5,5.5])],4)[0],[-1,0,0,1])
    # No valid aligned tokens must yield finite zero loss.
    centers=torch.full((3,2),-1.,device=dev)
    ti,ta,w=b.batch(['ACGT']*3,[np.ones(4)]*3)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        _,_,align,_,_=b.weighted_loss(frames,ti.to(dev),ta.to(dev),w.to(dev),centers,lens)
    assert torch.isfinite(align) and align==0
    print('PASS shared decoder init, short-frame boundary/length, centres, empty alignment',flush=True)


if __name__=='__main__':main()

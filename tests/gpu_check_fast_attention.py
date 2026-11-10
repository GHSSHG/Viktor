"""CUDA regression: FAST sliding FlashAttention vs masked SDPA, including padding.

Run: PYTHONPATH=. python tests/gpu_check_fast_attention.py
Uses a small batch; no checkpoints or persistent outputs are produced.
"""
import copy
import torch
import viktor.fast_encoder as fe


def main():
    assert torch.cuda.is_available() and fe.flash_attn_varlen_func is not None
    torch.manual_seed(17)
    model = fe.FastEncoder().cuda()
    ref = copy.deepcopy(model)
    x = torch.randn(4, 1, 1536, device="cuda")
    lengths = torch.tensor([1536, 960, 48, 0], device="cuda")
    weight = torch.randn(4, 32, 512, device="cuda")
    flash = fe.flash_attn_varlen_func

    def run(m):
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = m(x, lengths)
            loss = (y.float() * weight).sum() / weight.numel()
        loss.backward()
        return y.detach(), {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}

    fe.flash_attn_varlen_func = None
    expected, grads = run(ref)
    fe.flash_attn_varlen_func = flash
    actual, flash_grads = run(model)
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.03)
    for n in grads:
        assert torch.isfinite(flash_grads[n]).all(), n
        err = (flash_grads[n] - grads[n]).float().norm() / grads[n].float().norm().clamp_min(1e-8)
        # Tiny QK-norm gradients are sensitive to bf16 kernel rounding;
        # constrain their absolute error as well as larger gradients' norm error.
        delta = (flash_grads[n] - grads[n]).abs().max()
        assert err < 0.08 or delta < 2e-6, (n, err.item(), delta.item())
    assert actual[1, 20:].count_nonzero() == 0
    assert actual[2, 1:].count_nonzero() == 0
    assert actual[3].count_nonzero() == 0
    # Changes beyond each real end cannot affect its real frames.
    noisy = x.clone()
    for i, length in enumerate(lengths.tolist()):
        noisy[i, :, length:] = 100 * torch.randn_like(noisy[i, :, length:])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.testing.assert_close(model(noisy, lengths), actual, atol=0, rtol=0)
    compiled = torch.compile(model, dynamic=False)
    cy, cg = run(compiled)
    torch.testing.assert_close(cy, actual, atol=0.05, rtol=0.03)
    assert all(torch.isfinite(g).all() for g in cg.values())
    print("PASS: full-encoder outputs, parameter gradients, padding isolation, compiled backward")


if __name__ == "__main__":
    main()

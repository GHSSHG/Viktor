"""Dorado koi kernels for the encoder's transformer stack (JIT-built C++ extension).

Needs libkoi (the prebuilt Dorado dependency, from the ONT CDN):
  libkoi-0.7.5-Linux-x86_64-cuda-12.8.tar.gz, unpacked under third_party/ or at $VIKTOR_KOI_DIR.
"""

import glob
import os

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ext = None


def _koi_dir():
    d = os.environ.get("VIKTOR_KOI_DIR")
    if d:
        return d
    found = sorted(glob.glob(os.path.join(_ROOT, "third_party", "libkoi-*")))
    if not found:
        raise FileNotFoundError("libkoi not found: set VIKTOR_KOI_DIR or unpack it under third_party/")
    return found[-1]


def _cuda_runtime_dir():
    import nvidia.cuda_runtime
    return os.path.dirname(nvidia.cuda_runtime.__file__)


def load_extension():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        koi, cudart = _koi_dir(), _cuda_runtime_dir()
        _ext = load(
            name="viktor_koi_tx",
            sources=[os.path.join(_ROOT, "viktor", "csrc", "koi_tx.cpp")],
            extra_include_paths=[os.path.join(koi, "include"), os.path.join(cudart, "include")],
            extra_cflags=["-O3"],
            extra_ldflags=[os.path.join(koi, "lib", "libkoi.a"), f"-L{os.path.join(cudart, 'lib')}",
                           "-l:libcudart.so.12", f"-Wl,-rpath,{os.path.join(cudart, 'lib')}"],
            verbose=False,
        )
    return _ext


class KoiTxStack:
    """Replaces encoder.transformer_encoder; built from its (fp16, CUDA) weights."""

    def __init__(self, transformer_encoder, nhead, dim_feedforward, attn_window, theta=10000.0,
                 max_seq_len=4096, use_i8=True, remove_bits=4):
        ext = load_extension()
        layers, alphas = [], []
        for layer in transformer_encoder:
            a, f = layer.self_attn, layer.ff
            layers.append([t.detach().half().contiguous() for t in (
                a.Wqkv.weight, a.out_proj.weight, a.out_proj.bias, layer.norm1.weight,
                layer.norm2.weight, f.fc1.weight, f.fc2.weight)])
            alphas.append(float(layer.deepnorm_alpha))
        self.stack = ext.KoiTxStack(layers, alphas, nhead, dim_feedforward, int(attn_window[0]),
                                    int(attn_window[1]), float(theta), int(max_seq_len), use_i8,
                                    int(remove_bits))

    def __call__(self, x):
        return self.stack.forward(x.half().contiguous())

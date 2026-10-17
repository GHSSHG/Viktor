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


def _nvidia_dir(name):
    import nvidia
    return os.path.join(os.path.dirname(nvidia.__file__), name)


def load_extension():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        koi, cudart = _koi_dir(), _nvidia_dir("cuda_runtime")
        _ext = load(
            name="viktor_koi_tx",
            sources=[os.path.join(_ROOT, "viktor", "csrc", "koi_tx.cpp")],
            extra_include_paths=[os.path.join(koi, "include"), os.path.join(cudart, "include"),
                                 os.path.join(_nvidia_dir("cuda_nvcc"), "include")],
            extra_cflags=["-O3"],
            extra_ldflags=[os.path.join(koi, "lib", "libkoi.a"), f"-L{os.path.join(cudart, 'lib')}",
                           "-l:libcudart.so.12", f"-Wl,-rpath,{os.path.join(cudart, 'lib')}"],
            verbose=False,
        )
    return _ext


class KoiVcsEncoder:
    """Dorado's variable-chunk-size encoder path (conv stack + transformer) on koi, from the
    (fp16, CUDA) weights of a DoradoEncoder."""

    def __init__(self, encoder, use_i8=True, remove_bits=4):
        ext = load_extension()
        lay = encoder.tx_layer
        convs, params = [], []
        for c in encoder.conv:
            convs.append([c.conv.weight.detach().half().contiguous(), c.conv.bias.detach().half().contiguous()])
            params.append([c.conv.kernel_size[0], c.conv.stride[0]])
        layers, alphas = [], []
        for layer in encoder.transformer_encoder:
            a, f = layer.self_attn, layer.ff
            layers.append([t.detach().half().contiguous() for t in (
                a.Wqkv.weight, a.out_proj.weight, a.out_proj.bias, layer.norm1.weight,
                layer.norm2.weight, f.fc1.weight, f.fc2.weight)])
            alphas.append(float(layer.deepnorm_alpha))
        self.enc = ext.KoiVcsEncoder(convs, params, layers, alphas, lay["nhead"], lay["dim_feedforward"],
                                     float(lay.get("theta", 10000.0)), int(lay.get("max_seq_len", 4096)),
                                     int(encoder.chunksize), use_i8, int(remove_bits))
        self.granularity = self.enc.granularity()

    def __call__(self, signal, lens):
        """signal: 1D fp16 CUDA (chunks concatenated), lens: samples per chunk -> frames [sum/stride, C]."""
        return self.enc.forward(signal.half().contiguous(), [int(n) for n in lens])

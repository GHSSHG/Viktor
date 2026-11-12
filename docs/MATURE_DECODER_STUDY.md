# Mature encoder + small decoder (`scripts/train_mature.py`)

Two runs, both starting from the encoder of `runs/k4ft2/snap_0700000.pt`, each with a new projector and a new 4-layer
decoder (d512, FFN 1536, GQA 8/2, 4-mer tokens). The decoder initialisation is identical in both runs.

| run | prefix to the decoder | RoPE position scale |
|---|---|---|
| `mature_1x_decoder4_norecompute_b32` | Dorado frames, 12x (1024 per chunk) | 1 |
| `mature_4x_decoder4_norecompute_b32` | 4x compressed, 48x (256 per chunk) | 4 |

Compression (`viktor/mature_encoder.py`, 4x only): two stages, each concatenates adjacent frame pairs
(512 -> 1024 -> 2048 channels), then a depthwise k7 conv block and one sliding-window transformer layer
(attention width 512 = 8 x 64, SwiGLU 1024, window +-64 / +-32). Padding is masked after every stage.

## Training

- 2 GPUs per run, 32 windows per GPU, 200k steps. Data mix as `runs/k4ft2`.
- Steps 0-20k: the Dorado encoder is frozen. New modules lr 2e-4, warmup 2k.
- From 20k: the Dorado encoder trains too, lr 1e-5, warmup 2k. No activation recompute.
- WSD: constant to 160k, `1 - sqrt` decay over 160k-200k (`predecay_160000.pt` kept).
- Loss: CE + 0.1 x alignment KL (layers 1 and 3, heads 0 and 4; sigma 1 at the run's frame rate).
- AdamW (0.9, 0.95), weight decay 0.1 on matrices; one global clip at 1.0; bf16 autocast.

Launch: `COMPRESSION=1 GPUS=0,1 bash scripts/mature_study.sh` (or `COMPRESSION=4 GPUS=2,3`; `OUT=` for another run dir).

## Inference

`scripts/basecall_viktor.py` reads `encoder_arch == "mature"` from the snapshot: the Dorado layers run on koi
(`--encoder koi`, default) or PyTorch (`--encoder torch`, what training used); the compression runs in PyTorch on the
whole batch.

GPU check: `PYTHONPATH=. python tests/gpu_check_mature.py` (compression attention vs SDPA, padding isolation,
batched vs single-chunk encoding).

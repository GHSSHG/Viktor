# Mature Encoder → new small Decoder study

Approved configuration, 2026-11-11 (Asia/Shanghai). Both runs start from the
Encoder in `runs/k4ft2/snap_0700000.pt`; the projector and 4-layer Decoder are
new. Common Decoder parameters have architecture-independent initial seeds.

| Run | GPUs | Prefix | Decoder |
|---|---|---|---|
| `mature_1x_decoder4` | 0,1 | original 12x / 512 | 4 layers, d512, FFN1536, GQA 8/2 |
| `mature_4x_decoder4` | 2,3 | compressed 48x / 2048 → projector512 | identical |

The compressed arm has two pair-concatenation stages. Each has a depthwise
kernel7 convolution and dense residual projection, followed by one local
Transformer: residual width1024/2048, attention internal width512 (8x64),
SwiGLU intermediate1024, windows ±64/±32. Transformer parameters ~15.7M;
including convolution blocks, total compression parameters21,008,640.
Projector is Linear→GELU→Linear, 512→512→512 or 2048→512→512.
All Encoder attention requires CUDA sliding FlashAttention, with explicit
failure instead of falling back to dense masked SDPA.

## Optimization

- Two ranks per run, microbatch16/rank, accumulation2: global batch64.
- BF16 autocast, FP32 parameters, AdamW(.9,.95), decay.1 on matrices/conv,
  zero on norms/bias/embedding; one global gradient clip at1.0.
- New modules LR2e-4, 2k warmup, constant through160k, then40k WSD
  `1-sqrt(progress)`, total200k optimizer updates.
- Mature Encoder frozen/eval through20k; then added optimizer groups,
  LR1e-5 with2k warmup and same final WSD factor. Existing Adam states retained.
- Encoder eager/FlashAttention, activation checkpointing during joint phase.
  Launcher uses `--no-compile`: the server's Inductor/Triton compiler fails
  compiling Decoder loss even with split reductions disabled. FlashAttention
  and fused inference kernels remain enabled. Rebuild DDP at unfreezing.
- CE + .1 alignment KL. Supervise Decoder layers1,3 (zero-based), heads0,4.
  Sigma4 at12x, sigma1 at48x, centre `(c12+.5)/compression-.5`, clamped at0
  for the first frame. Missing alignments ignored; empty alignment batches=0.
- CE and KL each use global denominators over all ranks and accumulation
  microbatches, rather than averaging means with unequal valid-label counts.
- Same data proportions as `scripts/fast_ablation.sh`. Validation uses the same
  fixed windows. Streaming resumes are reseeded from the saved step, following
  the existing trainer convention; they do not replay an identical data stream.

## Evaluation and checkpoints

Every10k: fixed256 validation windows/source (CE, token ACC).
Training logs include alignment KL; fixed validation windows carry no forced
alignment centres, so validation KL is not fabricated.
Every20k: full resumable `last.pt` and fixed512 part2 heldout chunks, using the
existing NW reference-denominator identity (unclamped), median/mean/P10,
<90% fraction, and p99 predicted/reference length ratio. Chunk metrics are
not whole-read identity. Heldout decoding is sharded over both ranks.
Snapshots:20k/80k/160k/200k; full pre-decay state160k. Save atomically before
heldout evaluation, so decode errors do not discard completed training.
Final external human298/bacteria593 and component/end-to-end speed evaluations
are to be performed on completed checkpoints; no new automatic scheduler job.

Launch/resume:

```sh
COMPRESSION=1 GPUS=0,1 bash scripts/mature_study.sh
COMPRESSION=4 GPUS=2,3 bash scripts/mature_study.sh
```

Regression: `tests/gpu_check_mature.py` compares true FlashAttention against
SDPA outputs/gradients and checks padding, short frames, common initialization,
centre conversion, and empty alignment. Two-rank real-data smoke additionally
exercises accumulation, freeze/unfreeze, genuine Adam updates, rank agreement,
validation, decoding, checkpoint loading/resume. Temporary run directories
are removed after checks.

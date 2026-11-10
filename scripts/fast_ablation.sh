#!/usr/bin/env bash
# Four independent runs; launch one per GPU with VARIANT=A/AB/AC/ABC and GPU=0/1/2/3.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/myenv/bin/activate"
variant=${VARIANT:?set VARIANT} gpu=${GPU:?set GPU}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu
export OMP_NUM_THREADS=2
out="runs/fast_ablation_${variant}"
mkdir -p "$out"
cp -n runs/k4ft2/dorado.json "$out/" || true
bd="$HOME/Work/BasecallerData"
exec torchrun --standalone --nproc-per-node=1 -m scripts.train \
  --model-dir "$HOME/tools/dorado_models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0" \
  --out "$out" --encoder-arch fast --fast-variant "$variant" \
  --tokenizer 4mer --layers 8 --init-decoder runs/k4ft2/snap_0700000.pt \
  --batch 64 --seed 0 --workers 4 \
  --lr 1e-3 --warmup 2000 --freeze-decoder-steps 10000 \
  --decoder-lr 1e-4 --decoder-warmup 2000 \
  --max-steps 200000 --decay-start 160000 --decay-steps 40000 \
  --ctc-weight 0.5 --ctc-weight-joint 0.3 --ctc48-weight 0.5 --ctc48-weight-joint 0.5 \
  --prefix-pos-scale 4 --align-sigma 1.0 \
  --val-every 10000 --eval-n 256 --heldout-n 64 --ckpt-every 20000 --snapshot-every 40000 \
  --windows "$bd/viktor_win12288_part2:part2:0.1816" \
  /data/viktor_win_v3h:part1:0.6612 /data/viktor_win_v3x:part1:0.0073 \
  "$bd/viktor_win12288_bact:bact:0.0398" "$bd/viktor_win12288_hall:hall:0.1102"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/myenv/bin/activate"
compression=${COMPRESSION:?set COMPRESSION=1 or 4}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPUS:?set GPUS}
export OMP_NUM_THREADS=2
out=${OUT:-runs/mature_${compression}x_decoder4}
bd="$HOME/Work/BasecallerData"
exec torchrun --standalone --nproc-per-node=2 -m scripts.train_mature \
 --model-dir "$HOME/tools/dorado_models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0" \
 --encoder-init runs/k4ft2/snap_0700000.pt --compression "$compression" --out "$out" --no-compile \
 --windows "$bd/viktor_win12288_part2:part2:0.1816" \
 /data/viktor_win_v3h:part1:0.6612 /data/viktor_win_v3x:part1:0.0073 \
 "$bd/viktor_win12288_bact:bact:0.0398" "$bd/viktor_win12288_hall:hall:0.1102" "$@"

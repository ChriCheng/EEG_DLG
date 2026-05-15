#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# MI2 task-only debug sweep, round 2
#
# Round 1 showed:
#   - multiscale EEGNet peaks early around 51% BCA, then overfits
#   - classic EEGNet reaches similar BCA at 15 epochs but is still underfitting
#
# This sweep checks whether longer classic training and Euclidean Alignment are
# the main knobs before returning to full two-head training.
# ==============================================================================

python_bin="/home/ubuntu/miniconda3/envs/PP310/bin/python"

classic_args=(
  -m scripts.train
  --dataset MI2
  --model EEGNet
  --task_only
  --task_epochs 60
  --user_epochs 0
  --batch_size 8
  --weight_decay 0
  --seeds 0
  --normalize channel
  --save_every 0
  --device cuda
  --eegnet_F1 8
  --eegnet_D 2
  --eegnet_F2 16
  --eegnet_dropout 0.5
  --eegnet_temporal_kernels none
)

echo
echo "================================================================================"
echo "[MI2 debug round2] classic EEGNet + EA, lr=2e-3, 60 task epochs"
echo "================================================================================"
"$python_bin" "${classic_args[@]}" \
  --task_lr 2e-3 \
  --euclidean_align \
  --run_name mi2_debug_classic_ea_lr2e3_60ep_gpu

echo
echo "================================================================================"
echo "[MI2 debug round2] classic EEGNet + EA, lr=1e-3, 60 task epochs"
echo "================================================================================"
"$python_bin" "${classic_args[@]}" \
  --task_lr 1e-3 \
  --euclidean_align \
  --run_name mi2_debug_classic_ea_lr1e3_60ep_gpu

echo
echo "================================================================================"
echo "[MI2 debug round2] classic EEGNet without EA, lr=2e-3, 60 task epochs"
echo "================================================================================"
"$python_bin" "${classic_args[@]}" \
  --task_lr 2e-3 \
  --run_name mi2_debug_classic_noea_lr2e3_60ep_gpu

echo
echo "================================================================================"
echo "[MI2 debug round2] multiscale EEGNet without EA, lr=2e-3, 30 task epochs"
echo "================================================================================"
"$python_bin" -m scripts.train \
  --dataset MI2 \
  --model EEGNet \
  --task_only \
  --task_epochs 30 \
  --user_epochs 0 \
  --batch_size 8 \
  --task_lr 2e-3 \
  --weight_decay 0 \
  --seeds 0 \
  --normalize channel \
  --save_every 0 \
  --device cuda \
  --run_name mi2_debug_ms_noea_lr2e3_30ep_gpu

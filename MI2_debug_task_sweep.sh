#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# MI2 task-only debug sweep
#
# Purpose:
#   Fast task/test BCA comparison for MI2 without training the user head.
#
# Requires the local scripts.train debug flags:
#   --task_only
#   --eegnet_F1 / --eegnet_D / --eegnet_F2 / --eegnet_dropout
#   --eegnet_temporal_kernels none
# ==============================================================================

python_bin="/home/ubuntu/miniconda3/envs/PP310/bin/python"

common_args=(
  -m scripts.train
  --dataset MI2
  --model EEGNet
  --task_only
  --task_epochs 15
  --user_epochs 0
  --batch_size 8
  --task_lr 2e-3
  --weight_decay 0
  --seeds 0
  --normalize channel
  --euclidean_align
  --save_every 0
  --device cuda
)

echo
echo "================================================================================"
echo "[MI2 debug] multiscale EEGNet, 15 task epochs"
echo "================================================================================"
"$python_bin" "${common_args[@]}" \
  --run_name mi2_debug_ms_15ep_gpu

echo
echo "================================================================================"
echo "[MI2 debug] classic EEGNet, 15 task epochs"
echo "================================================================================"
"$python_bin" "${common_args[@]}" \
  --run_name mi2_debug_classic_15ep_gpu \
  --eegnet_F1 8 \
  --eegnet_D 2 \
  --eegnet_F2 16 \
  --eegnet_dropout 0.5 \
  --eegnet_temporal_kernels none

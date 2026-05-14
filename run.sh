#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run 参数填写区
#
# P300 two-stage baseline: first train task head, then train user head.
# ==============================================================================
python_bin="/home/ubuntu/miniconda3/envs/PP310/bin/python"

dataset="P300"
model="EEGNet"
task_epochs="100"
user_epochs="300"
batch_size="8"
task_lr="2e-3"
user_lr="2e-3"
weight_decay="0"
seeds="0"
normalize="channel"
euclidean_align="1"
task_balanced_sampler="1"
run_name="p300_eegnet_channel"

extra_args=()
if [[ "$euclidean_align" == "1" || "$euclidean_align" == "true" ]]; then
  extra_args+=(--euclidean_align)
fi
if [[ "$task_balanced_sampler" == "1" || "$task_balanced_sampler" == "true" ]]; then
  extra_args+=(--task_balanced_sampler)
fi

echo
echo "================================================================================"
echo "[run] dataset=$dataset model=$model run_name=$run_name"
echo "[run] task_epochs=$task_epochs user_epochs=$user_epochs batch_size=$batch_size seeds=$seeds"
echo "[run] normalize=$normalize euclidean_align=$euclidean_align task_balanced_sampler=$task_balanced_sampler"
echo "================================================================================"

exec "$python_bin" -u -m scripts.train \
  --dataset "$dataset" \
  --model "$model" \
  --task_epochs "$task_epochs" \
  --user_epochs "$user_epochs" \
  --batch_size "$batch_size" \
  --task_lr "$task_lr" \
  --user_lr "$user_lr" \
  --weight_decay "$weight_decay" \
  --seeds "$seeds" \
  --normalize "$normalize" \
  --run_name "$run_name" \
  "${extra_args[@]}"

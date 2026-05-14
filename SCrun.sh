#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# SCrun 参数填写区
#
# ShallowCNN two-stage training.
# ==============================================================================
python_bin="/home/ubuntu/miniconda3/envs/PP310/bin/python"

dataset="P300"
model="ShallowCNN"
task_epochs="150"
user_epochs="150"
batch_size="128"
task_lr="1e-2"
user_lr="1e-2"
weight_decay="5e-4"
seeds="0"
normalize="channel"
user_hidden_dim="50"
user_dropout="0"
save_root="checkpoint/checkpoints_2stage"
run_name="p300_shallowcnn_ea_channel_paper"
device="auto"
euclidean_align="1"
task_balanced_sampler="1"

extra_args=()
if [[ "$euclidean_align" == "1" || "$euclidean_align" == "true" ]]; then
  extra_args+=(--euclidean_align)
fi
if [[ "$task_balanced_sampler" == "1" || "$task_balanced_sampler" == "true" ]]; then
  extra_args+=(--task_balanced_sampler)
fi

echo
echo "================================================================================"
echo "[SCrun] dataset=$dataset model=$model run_name=$run_name"
echo "[SCrun] task_epochs=$task_epochs user_epochs=$user_epochs batch_size=$batch_size seeds=$seeds"
echo "[SCrun] normalize=$normalize euclidean_align=$euclidean_align task_balanced_sampler=$task_balanced_sampler"
echo "[SCrun] save_root=$save_root device=$device"
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
  --user_hidden_dim "$user_hidden_dim" \
  --user_dropout "$user_dropout" \
  --save_root "$save_root" \
  --run_name "$run_name" \
  --device "$device" \
  "${extra_args[@]}"

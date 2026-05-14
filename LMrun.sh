#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# LMrun 参数填写区
#
# LMEEGNet freeze/two-stage training.
# ==============================================================================
python_bin="/home/ubuntu/miniconda3/envs/PP310/bin/python"

dataset="P300"
model="LMEEGNet"
tag="ea"
task_epochs="100"
user_epochs="100"
batch_size="8"
task_lr="2e-3"
user_lr="2e-3"
weight_decay="1e-4"
seeds="0"
normalize="none"
save_root="checkpoint/checkpoints_LMEEGNet"
euclidean_align="1"
task_balanced_sampler="1"
run_name="${dataset}_${model}_${tag}"

extra_args=()
if [[ "$euclidean_align" == "1" || "$euclidean_align" == "true" ]]; then
  extra_args+=(--euclidean_align)
fi
if [[ "$task_balanced_sampler" == "1" || "$task_balanced_sampler" == "true" ]]; then
  extra_args+=(--task_balanced_sampler)
fi

echo
echo "================================================================================"
echo "[LMrun] dataset=$dataset model=$model tag=$tag run_name=$run_name"
echo "[LMrun] task_epochs=$task_epochs user_epochs=$user_epochs batch_size=$batch_size seeds=$seeds"
echo "[LMrun] normalize=$normalize euclidean_align=$euclidean_align task_balanced_sampler=$task_balanced_sampler"
echo "[LMrun] save_root=$save_root"
echo "================================================================================"

exec "$python_bin" -u -m scripts.train_freeze \
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
  --save_root "$save_root" \
  --run_name "$run_name" \
  "${extra_args[@]}"

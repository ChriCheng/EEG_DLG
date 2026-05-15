#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# MI2_run_leave1 参数填写区
#
# UIP-EEG 论文命名下的 MI2:
#   BCI Competition IV Dataset 2a / BNCI2014_001
#   9 users, 22 channels, 4 MI classes, 2 sessions.
#
# 双头训练沿用 scripts.train:
#   Stage 1: task classifier
#   Stage 2: user classifier
# Leave-one-session-out 会分别用一个 session 训练、另一个 session 测试。
# ==============================================================================
python_bin="/home/ubuntu/miniconda3/envs/PP310/bin/python"

dataset="MI2"
model="EEGNet"
task_epochs="60"
user_epochs="60"
batch_size="8"
task_lr="2e-3"
user_lr="2e-3"
weight_decay="0"
seeds="0"
normalize="channel"
euclidean_align="1"
task_balanced_sampler="0"
eegnet_F1="8"
eegnet_D="2"
eegnet_F2="16"
eegnet_dropout="0.5"
eegnet_temporal_kernels="none"
run_name="mi2_classic_ea_lr2e3_60ep_leave1"

extra_args=()
if [[ "$euclidean_align" == "1" || "$euclidean_align" == "true" ]]; then
  extra_args+=(--euclidean_align)
fi
if [[ "$task_balanced_sampler" == "1" || "$task_balanced_sampler" == "true" ]]; then
  extra_args+=(--task_balanced_sampler)
fi

echo
echo "================================================================================"
echo "[MI2_run_leave1] dataset=$dataset model=$model"
echo "[MI2_run_leave1] task_epochs=$task_epochs user_epochs=$user_epochs batch_size=$batch_size seeds=$seeds"
echo "[MI2_run_leave1] normalize=$normalize euclidean_align=$euclidean_align task_balanced_sampler=$task_balanced_sampler"
echo "[MI2_run_leave1] EEGNet classic: F1=$eegnet_F1 D=$eegnet_D F2=$eegnet_F2 dropout=$eegnet_dropout temporal_kernels=$eegnet_temporal_kernels"
echo "[MI2_run_leave1] run_name=$run_name"
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
  --eegnet_F1 "$eegnet_F1" \
  --eegnet_D "$eegnet_D" \
  --eegnet_F2 "$eegnet_F2" \
  --eegnet_dropout "$eegnet_dropout" \
  --eegnet_temporal_kernels "$eegnet_temporal_kernels" \
  "${extra_args[@]}"

#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# MI2_DLG_run_leave1 参数填写区
#
# Uses data/MI2_DLG, where the last 88 trials from each original MI2 session
# are relabeled as artificial session 3 for DLG holdout.
#
# Effective split:
#   session 1 -> Stage 1 task training
#   session 2 -> Stage 1 task test + Stage 2 user-head training
#   session 3 -> Stage 2 user-head eval + DLG holdout
# ==============================================================================
python_bin="/home/ubuntu/miniconda3/envs/PP310/bin/python"

dataset="MI2_DLG"
model="EEGNet"
holdout_session="3"
task_train_session="1"
user_train_split="test"
user_eval_split="holdout"
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
run_name="mi2_dlg_s1task_s2user_usereval_s3_classic_ea_lr2e3_60ep"

extra_args=()
if [[ "$euclidean_align" == "1" || "$euclidean_align" == "true" ]]; then
  extra_args+=(--euclidean_align)
fi
if [[ "$task_balanced_sampler" == "1" || "$task_balanced_sampler" == "true" ]]; then
  extra_args+=(--task_balanced_sampler)
fi

echo
echo "================================================================================"
echo "[MI2_DLG_run_leave1] dataset=$dataset model=$model task_train_session=$task_train_session holdout_session=$holdout_session"
echo "[MI2_DLG_run_leave1] session split: session 1 task train, session 2 task test + user-head train, session 3 user-head eval + DLG holdout"
echo "[MI2_DLG_run_leave1] task_epochs=$task_epochs user_epochs=$user_epochs batch_size=$batch_size seeds=$seeds"
echo "[MI2_DLG_run_leave1] normalize=$normalize euclidean_align=$euclidean_align task_balanced_sampler=$task_balanced_sampler"
echo "[MI2_DLG_run_leave1] EEGNet classic: F1=$eegnet_F1 D=$eegnet_D F2=$eegnet_F2 dropout=$eegnet_dropout temporal_kernels=$eegnet_temporal_kernels"
echo "[MI2_DLG_run_leave1] run_name=$run_name"
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
  --dlg_holdout_session_original "$holdout_session" \
  --task_train_session_original "$task_train_session" \
  --user_train_split "$user_train_split" \
  --user_eval_split "$user_eval_split" \
  --run_name "$run_name" \
  --eegnet_F1 "$eegnet_F1" \
  --eegnet_D "$eegnet_D" \
  --eegnet_F2 "$eegnet_F2" \
  --eegnet_dropout "$eegnet_dropout" \
  --eegnet_temporal_kernels "$eegnet_temporal_kernels" \
  "${extra_args[@]}"

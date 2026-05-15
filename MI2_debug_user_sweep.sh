#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# MI2_DLG user-head debug sweep
#
# Fixed split:
#   session 1 -> Stage 1 task training
#   session 2 -> Stage 1 task test + Stage 2 user-head training
#   session 3 -> Stage 2 user-head eval + DLG holdout
#
# Purpose:
#   Current full run gets task BCA around 52.8%, but user-head holdout UIA only
#   reaches about 44% while train UIA climbs past 95%. This sweep keeps the best
#   task-side classic EEGNet setup and tests smaller/slower/stronger-dropout
#   user heads to reduce session-2 overfitting.
# ==============================================================================

python_bin="/home/ubuntu/miniconda3/envs/PP310/bin/python"

common_args=(
  -m scripts.train
  --dataset MI2_DLG
  --model EEGNet
  --task_train_session_original 1
  --dlg_holdout_session_original 3
  --user_train_split test
  --user_eval_split holdout
  --task_epochs 60
  --user_epochs 80
  --batch_size 8
  --task_lr 2e-3
  --weight_decay 0
  --seeds 0
  --normalize channel
  --euclidean_align
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
echo "[MI2 user debug] baseline user head: hidden=256 dropout=0.5 lr=2e-3"
echo "================================================================================"
"$python_bin" "${common_args[@]}" \
  --user_hidden_dim 256 \
  --user_dropout 0.5 \
  --user_lr 2e-3 \
  --run_name mi2_debug_user_h256_d05_lr2e3

echo
echo "================================================================================"
echo "[MI2 user debug] lower user lr: hidden=256 dropout=0.5 lr=1e-3"
echo "================================================================================"
"$python_bin" "${common_args[@]}" \
  --user_hidden_dim 256 \
  --user_dropout 0.5 \
  --user_lr 1e-3 \
  --run_name mi2_debug_user_h256_d05_lr1e3

echo
echo "================================================================================"
echo "[MI2 user debug] lower user lr: hidden=256 dropout=0.5 lr=5e-4"
echo "================================================================================"
"$python_bin" "${common_args[@]}" \
  --user_hidden_dim 256 \
  --user_dropout 0.5 \
  --user_lr 5e-4 \
  --run_name mi2_debug_user_h256_d05_lr5e4

echo
echo "================================================================================"
echo "[MI2 user debug] smaller user head: hidden=128 dropout=0.5 lr=1e-3"
echo "================================================================================"
"$python_bin" "${common_args[@]}" \
  --user_hidden_dim 128 \
  --user_dropout 0.5 \
  --user_lr 1e-3 \
  --run_name mi2_debug_user_h128_d05_lr1e3

echo
echo "================================================================================"
echo "[MI2 user debug] smaller user head: hidden=64 dropout=0.5 lr=1e-3"
echo "================================================================================"
"$python_bin" "${common_args[@]}" \
  --user_hidden_dim 64 \
  --user_dropout 0.5 \
  --user_lr 1e-3 \
  --run_name mi2_debug_user_h64_d05_lr1e3

echo
echo "================================================================================"
echo "[MI2 user debug] stronger dropout: hidden=128 dropout=0.7 lr=1e-3"
echo "================================================================================"
"$python_bin" "${common_args[@]}" \
  --user_hidden_dim 128 \
  --user_dropout 0.7 \
  --user_lr 1e-3 \
  --run_name mi2_debug_user_h128_d07_lr1e3


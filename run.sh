#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/ubuntu/miniconda3/envs/PP310/bin/python

# P300 two-stage baseline: task classifier, then user classifier.
# exec $PYTHON -m scripts.train \
#   --dataset P300 \
#   --model EEGNet \
#   --task_epochs 100 \
#   --user_epochs 100 \
#   --batch_size 8 \
#   --task_lr 2e-3 \
#   --user_lr 1e-3 \
#   --weight_decay 1e-4 \
#   --seeds 0 \
#   --normalize none \
#   --run_name p300_eegnet_base

# P300 two-stage baseline with subject-wise Euclidean Alignment.
exec $PYTHON -u -m scripts.train \
  --dataset P300 \
  --model EEGNet \
  --task_epochs 100 \
  --user_epochs 300 \
  --batch_size 8 \
  --task_lr 2e-3 \
  --user_lr 2e-3 \
  --weight_decay 0 \
  --seeds 0 \
  --normalize channel \
  --euclidean_align \
  --task_balanced_sampler \
  --run_name p300_eegnet_channel

# P300 user-only sanity check.
# $PYTHON -m scripts.train_user_only \
#   --mi1_dir data/P300 \
#   --epochs 100 \
#   --batch_size 8 \
#   --lr 2e-3 \
#   --user_hidden_dim 128 \
#   --user_dropout 0.5 \
#   --normalize none \
#   --weight_decay 1e-4 \
#   --seeds 0 \
#   --run_name p300_user_only_base

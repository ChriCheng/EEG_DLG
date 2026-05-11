#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/ubuntu/miniconda3/envs/PP310/bin/python}
DATASET=${DATASET:-P300}
SEEDS=${SEEDS:-0}
TASK_EPOCHS=${TASK_EPOCHS:-100}
USER_EPOCHS=${USER_EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-8}
TASK_LR=${TASK_LR:-2e-3}
USER_LR=${USER_LR:-2e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
NORMALIZE=${NORMALIZE:-none}
SAVE_ROOT=${SAVE_ROOT:-checkpoint/checkpoints_LMEEGNet}

run_one() {
  local model="$1"
  local tag="$2"
  shift 2

  echo
  echo "================================================================================"
  echo "[LMrun] dataset=${DATASET} model=${model} tag=${tag} normalize=${NORMALIZE}"
  echo "================================================================================"

  "exec $PYTHON" -u -m scripts.train_freeze \
    --dataset "$DATASET" \
    --model "$model" \
    --task_epochs "$TASK_EPOCHS" \
    --user_epochs "$USER_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --task_lr "$TASK_LR" \
    --user_lr "$USER_LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --seeds "$SEEDS" \
    --normalize "$NORMALIZE" \
    --save_root "$SAVE_ROOT" \
    --run_name "${DATASET}_${model}_${tag}" \
    "$@"
}

run_one LMEEGNet ea --euclidean_align

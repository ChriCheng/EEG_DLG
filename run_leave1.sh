#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/ubuntu/miniconda3/envs/PP310/bin/python}

DATASET=${DATASET:-P300}
MODEL=${MODEL:-EEGNet}
TASK_EPOCHS=${TASK_EPOCHS:-100}
USER_EPOCHS=${USER_EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-8}
TASK_LR=${TASK_LR:-2e-3}
USER_LR=${USER_LR:-2e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}
SEEDS=${SEEDS:-0}
NORMALIZE=${NORMALIZE:-channel}
EUCLIDEAN_ALIGN=${EUCLIDEAN_ALIGN:-1}
TASK_BALANCED_SAMPLER=${TASK_BALANCED_SAMPLER:-1}
HOLDOUT_SESSION=${HOLDOUT_SESSION:-4}
RUN_NAME=${RUN_NAME:-p300_eegnet_channel_leave_session_${HOLDOUT_SESSION}}

EXTRA_ARGS=()
if [[ "$EUCLIDEAN_ALIGN" == "1" || "$EUCLIDEAN_ALIGN" == "true" ]]; then
  EXTRA_ARGS+=(--euclidean_align)
fi
if [[ "$TASK_BALANCED_SAMPLER" == "1" || "$TASK_BALANCED_SAMPLER" == "true" ]]; then
  EXTRA_ARGS+=(--task_balanced_sampler)
fi

echo
echo "================================================================================"
echo "[run_leave1] dataset=${DATASET} model=${MODEL} holdout_session=${HOLDOUT_SESSION}"
echo "[run_leave1] task_epochs=${TASK_EPOCHS} user_epochs=${USER_EPOCHS} seeds=${SEEDS}"
echo "[run_leave1] run_name=${RUN_NAME}"
echo "================================================================================"

exec "$PYTHON" -u -m scripts.train \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --task_epochs "$TASK_EPOCHS" \
  --user_epochs "$USER_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --task_lr "$TASK_LR" \
  --user_lr "$USER_LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --seeds "$SEEDS" \
  --normalize "$NORMALIZE" \
  --dlg_holdout_session_original "$HOLDOUT_SESSION" \
  --run_name "$RUN_NAME" \
  "${EXTRA_ARGS[@]}"

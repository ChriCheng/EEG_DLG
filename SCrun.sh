#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/ubuntu/miniconda3/envs/PP310/bin/python}
DATASET=${DATASET:-P300}
SEEDS=${SEEDS:-0}
TASK_EPOCHS=${TASK_EPOCHS:-150}
USER_EPOCHS=${USER_EPOCHS:-150}
BATCH_SIZE=${BATCH_SIZE:-128}
TASK_LR=${TASK_LR:-1e-2}
USER_LR=${USER_LR:-1e-2}
WEIGHT_DECAY=${WEIGHT_DECAY:-5e-4}
NORMALIZE=${NORMALIZE:-channel}
USER_HIDDEN_DIM=${USER_HIDDEN_DIM:-50}
USER_DROPOUT=${USER_DROPOUT:-0}
SAVE_ROOT=${SAVE_ROOT:-checkpoint/checkpoints_2stage}
RUN_NAME=${RUN_NAME:-p300_shallowcnn_ea_channel_paper}
DEVICE=${DEVICE:-auto}
EUCLIDEAN_ALIGN=${EUCLIDEAN_ALIGN:-1}
TASK_BALANCED_SAMPLER=${TASK_BALANCED_SAMPLER:-1}

EXTRA_ARGS=()
if [[ "$EUCLIDEAN_ALIGN" == "1" || "$EUCLIDEAN_ALIGN" == "true" ]]; then
  EXTRA_ARGS+=(--euclidean_align)
fi
if [[ "$TASK_BALANCED_SAMPLER" == "1" || "$TASK_BALANCED_SAMPLER" == "true" ]]; then
  EXTRA_ARGS+=(--task_balanced_sampler)
fi

echo
echo "================================================================================"
echo "[SCrun] dataset=${DATASET} model=ShallowCNN normalize=${NORMALIZE} ea=${EUCLIDEAN_ALIGN} run=${RUN_NAME}"
echo "================================================================================"

exec "$PYTHON" -u -m scripts.train \
  --dataset "$DATASET" \
  --model ShallowCNN \
  --task_epochs "$TASK_EPOCHS" \
  --user_epochs "$USER_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --task_lr "$TASK_LR" \
  --user_lr "$USER_LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --seeds "$SEEDS" \
  --normalize "$NORMALIZE" \
  --user_hidden_dim "$USER_HIDDEN_DIM" \
  --user_dropout "$USER_DROPOUT" \
  --save_root "$SAVE_ROOT" \
  --run_name "$RUN_NAME" \
  --device "$DEVICE" \
  "${EXTRA_ARGS[@]}"

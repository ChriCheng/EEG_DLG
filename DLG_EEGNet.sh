#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/ubuntu/miniconda3/envs/PP310/bin/python}

DATASET=${DATASET:-P300}
MODEL=${MODEL:-EEGNet}
DATA_DIR=${DATA_DIR:-data/${DATASET}}
NORMALIZE=${NORMALIZE:-channel}
EUCLIDEAN_ALIGN=${EUCLIDEAN_ALIGN:-1}
DEVICE=${DEVICE:-auto}

SEED=${SEED:-0}
TRAIN_SESSION=${TRAIN_SESSION:-3}
CHECKPOINT=${CHECKPOINT:-checkpoint/checkpoints_2stage_EEGNet/p300_eegnet_channel/seed_${SEED}_train_session_${TRAIN_SESSION}/best_user_by_acc.pth}

BATCH_SIZE=${BATCH_SIZE:-1}
SPLIT=${SPLIT:-train}
EVAL_SESSION=${EVAL_SESSION:-}
ATTACK_HEAD=${ATTACK_HEAD:-task}
LABEL_MODE=${LABEL_MODE:-idlg}
ITERS=${ITERS:-33}
LR=${LR:-1.0}
OPTIMIZER=${OPTIMIZER:-lbfgs}
LOG_EVERY=${LOG_EVERY:-3}
TOPK=${TOPK:-3}
PLOT_CHANNEL=${PLOT_CHANNEL:--1}
SFREQ=${SFREQ:-128}

USER_HIDDEN_DIM=${USER_HIDDEN_DIM:-256}
USER_DROPOUT=${USER_DROPOUT:-0.5}
OUT_DIR=${OUT_DIR:-checkpoint/dlg_attack/${DATASET}_${MODEL}_${ATTACK_HEAD}_${LABEL_MODE}_seed${SEED}_session${TRAIN_SESSION}}

EXTRA_ARGS=()
if [[ "$EUCLIDEAN_ALIGN" == "1" || "$EUCLIDEAN_ALIGN" == "true" ]]; then
  EXTRA_ARGS+=(--euclidean_align)
fi
if [[ -n "$EVAL_SESSION" ]]; then
  EXTRA_ARGS+=(--eval_session_original "$EVAL_SESSION")
fi

echo
echo "================================================================================"
echo "[DLG_EEGNet] dataset=${DATASET} model=${MODEL} checkpoint=${CHECKPOINT}"
echo "[DLG_EEGNet] split=${SPLIT} eval_session=${EVAL_SESSION:-all} attack_head=${ATTACK_HEAD} label_mode=${LABEL_MODE} batch_size=${BATCH_SIZE} iters=${ITERS}"
echo "[DLG_EEGNet] out_dir=${OUT_DIR}"
echo "================================================================================"

exec "$PYTHON" -u -m scripts.dlg_attack \
  --checkpoint "$CHECKPOINT" \
  --dataset "$DATASET" \
  --data_dir "$DATA_DIR" \
  --model "$MODEL" \
  --normalize "$NORMALIZE" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --split "$SPLIT" \
  --batch_size "$BATCH_SIZE" \
  --attack_head "$ATTACK_HEAD" \
  --label_mode "$LABEL_MODE" \
  --iters "$ITERS" \
  --lr "$LR" \
  --optimizer "$OPTIMIZER" \
  --log_every "$LOG_EVERY" \
  --topk "$TOPK" \
  --plot_channel "$PLOT_CHANNEL" \
  --sfreq "$SFREQ" \
  --user_hidden_dim "$USER_HIDDEN_DIM" \
  --user_dropout "$USER_DROPOUT" \
  --out_dir "$OUT_DIR" \
  "${EXTRA_ARGS[@]}"

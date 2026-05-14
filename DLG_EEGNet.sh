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
EVAL_SESSION=${EVAL_SESSION:-}
if [[ -z "${TRAIN_SESSION:-}" ]]; then
  if [[ -n "$EVAL_SESSION" ]]; then
    TRAIN_SESSION=2
  else
    TRAIN_SESSION=3
  fi
fi
if [[ -z "${RUN_NAME:-}" ]]; then
  if [[ -n "$EVAL_SESSION" ]]; then
    RUN_NAME="p300_eegnet_channel_leave_session_${EVAL_SESSION}"
  else
    RUN_NAME="p300_eegnet_channel"
  fi
fi
FOLD_NAME=${FOLD_NAME:-seed_${SEED}_train_session_${TRAIN_SESSION}${EVAL_SESSION:+_holdout_session_${EVAL_SESSION}}}

BATCH_SIZE=${BATCH_SIZE:-1}
INDICES=${INDICES:-}
SPLIT=${SPLIT:-test}
ATTACK_HEAD=${ATTACK_HEAD:-task}
if [[ -z "${CHECKPOINT:-}" ]]; then
  CHECKPOINT=checkpoint/checkpoints_2stage_${MODEL}/${RUN_NAME}/${FOLD_NAME}/best_user_by_acc.pth
fi
LABEL_MODE=${LABEL_MODE:-idlg}
ITERS=${ITERS:-30}
LR=${LR:-1.0}
OPTIMIZER=${OPTIMIZER:-lbfgs}
LOG_EVERY=${LOG_EVERY:-2}
TOPK=${TOPK:-3}
PLOT_CHANNEL=${PLOT_CHANNEL:--1}
SFREQ=${SFREQ:-128}
WAVEFORM_GRID=${WAVEFORM_GRID:-0}
WAVEFORM_FONT_SIZE=${WAVEFORM_FONT_SIZE:-16}
EPSILON=${EPSILON:-0}
TRIAL_LAPLACE_SENSITIVITY=${TRIAL_LAPLACE_SENSITIVITY:-1.0}

USER_HIDDEN_DIM=${USER_HIDDEN_DIM:-256}
USER_DROPOUT=${USER_DROPOUT:-0.5}
EVAL_TAG=${EVAL_SESSION:-4}
INDICES_TAG=${INDICES:-random}
INDICES_TAG=${INDICES_TAG//,/p}
INDICES_TAG=${INDICES_TAG// /}
EPSILON_TAG=${EPSILON//./p}
EPSILON_TAG=${EPSILON_TAG//-/m}
EPSILON_TAG=${EPSILON_TAG//+/p}
SENSITIVITY_TAG=${TRIAL_LAPLACE_SENSITIVITY//./p}
SENSITIVITY_TAG=${SENSITIVITY_TAG//-/m}
SENSITIVITY_TAG=${SENSITIVITY_TAG//+/p}
DP_TAG="eps${EPSILON_TAG}_sens${SENSITIVITY_TAG}"
AUTO_OUT_DIR=0
if [[ -z "${OUT_DIR:-}" ]]; then
  AUTO_OUT_DIR=1
  OUT_DIR=checkpoint/dlg_attack/${DATASET}_${MODEL}_${ATTACK_HEAD}_${LABEL_MODE}_seed${SEED}_train${TRAIN_SESSION}_split${SPLIT}_eval${EVAL_TAG}_batch${BATCH_SIZE}_trial${INDICES_TAG}_${DP_TAG}
fi

EXTRA_ARGS=()
if [[ "$EUCLIDEAN_ALIGN" == "1" || "$EUCLIDEAN_ALIGN" == "true" ]]; then
  EXTRA_ARGS+=(--euclidean_align)
fi
if [[ -n "$EVAL_SESSION" ]]; then
  EXTRA_ARGS+=(--eval_session_original "$EVAL_SESSION")
fi
if [[ -n "$INDICES" ]]; then
  EXTRA_ARGS+=(--indices "$INDICES")
fi
if [[ "$AUTO_OUT_DIR" == "1" ]]; then
  EXTRA_ARGS+=(--append_indices_to_out_dir)
fi
if [[ "$EPSILON" != "0" && "$EPSILON" != "0.0" && -n "$EPSILON" ]]; then
  EXTRA_ARGS+=(--trial_laplace_epsilon "$EPSILON")
  EXTRA_ARGS+=(--trial_laplace_sensitivity "$TRIAL_LAPLACE_SENSITIVITY")
fi
if [[ "$WAVEFORM_GRID" == "0" || "$WAVEFORM_GRID" == "false" || "$WAVEFORM_GRID" == "False" || "$WAVEFORM_GRID" == "off" || "$WAVEFORM_GRID" == "no" ]]; then
  EXTRA_ARGS+=(--no-waveform_grid)
fi

echo
echo "================================================================================"
echo "[DLG_EEGNet] dataset=${DATASET} model=${MODEL} checkpoint=${CHECKPOINT}"
echo "[DLG_EEGNet] split=${SPLIT} eval_session=${EVAL_SESSION:-all} indices=${INDICES:-random} attack_head=${ATTACK_HEAD} label_mode=${LABEL_MODE} batch_size=${BATCH_SIZE} iters=${ITERS}"
echo "[DLG_EEGNet] plot_channel=${PLOT_CHANNEL} waveform_grid=${WAVEFORM_GRID} waveform_font_size=${WAVEFORM_FONT_SIZE}"
echo "[DLG_EEGNet] gradient_laplace_epsilon=${EPSILON} sensitivity=${TRIAL_LAPLACE_SENSITIVITY}"
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
  --waveform_font_size "$WAVEFORM_FONT_SIZE" \
  --user_hidden_dim "$USER_HIDDEN_DIM" \
  --user_dropout "$USER_DROPOUT" \
  --out_dir "$OUT_DIR" \
  "${EXTRA_ARGS[@]}"

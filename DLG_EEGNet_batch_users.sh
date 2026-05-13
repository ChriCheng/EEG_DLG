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
SPLIT=${SPLIT:-test}
EVAL_SESSION=${EVAL_SESSION:-}
SELECTION=${SELECTION:-all}
if [[ -z "${RUN_NAME:-}" ]]; then
  if [[ -n "$EVAL_SESSION" ]]; then
    RUN_NAME="p300_eegnet_channel_leave_session_${EVAL_SESSION}"
  else
    RUN_NAME="p300_eegnet_channel"
  fi
fi
FOLD_NAME=${FOLD_NAME:-seed_${SEED}_train_session_${TRAIN_SESSION}${EVAL_SESSION:+_holdout_session_${EVAL_SESSION}}}
CHECKPOINT=${CHECKPOINT:-checkpoint/checkpoints_2stage_${MODEL}/${RUN_NAME}/${FOLD_NAME}/best_user_by_acc.pth}
ATTACK_HEAD=${ATTACK_HEAD:-task}
LABEL_MODE=${LABEL_MODE:-idlg}
ITERS=${ITERS:-33}
LR=${LR:-1.0}
OPTIMIZER=${OPTIMIZER:-lbfgs}
LOG_EVERY=${LOG_EVERY:-3}
TOPK=${TOPK:-3}
SKIP_FIGURES=${SKIP_FIGURES:-1}
KEEP_TRIAL_ARTIFACTS=${KEEP_TRIAL_ARTIFACTS:-0}
MAX_TRIALS=${MAX_TRIALS:-0}
START_OFFSET=${START_OFFSET:-0}

USER_HIDDEN_DIM=${USER_HIDDEN_DIM:-256}
USER_DROPOUT=${USER_DROPOUT:-0.5}
PLOT_CHANNEL=${PLOT_CHANNEL:--1}
SFREQ=${SFREQ:-128}
OUT_DIR=${OUT_DIR:-checkpoint/dlg_attack_batch/${DATASET}_${MODEL}_${ATTACK_HEAD}_${LABEL_MODE}_seed${SEED}_train${TRAIN_SESSION}_eval${EVAL_SESSION:-all}_${SELECTION}}

EXTRA_ARGS=()
if [[ "$EUCLIDEAN_ALIGN" == "1" || "$EUCLIDEAN_ALIGN" == "true" ]]; then
  EXTRA_ARGS+=(--euclidean_align)
fi
if [[ -n "$EVAL_SESSION" ]]; then
  EXTRA_ARGS+=(--eval_session_original "$EVAL_SESSION")
fi
if [[ "$SKIP_FIGURES" == "1" || "$SKIP_FIGURES" == "true" ]]; then
  EXTRA_ARGS+=(--skip_figures)
fi
if [[ "$KEEP_TRIAL_ARTIFACTS" == "1" || "$KEEP_TRIAL_ARTIFACTS" == "true" ]]; then
  EXTRA_ARGS+=(--keep_trial_artifacts)
fi

echo
echo "================================================================================"
echo "[DLG_EEGNet_batch_users] dataset=${DATASET} model=${MODEL} checkpoint=${CHECKPOINT}"
echo "[DLG_EEGNet_batch_users] selection=${SELECTION} | split=${SPLIT} eval_session=${EVAL_SESSION:-all} attack_head=${ATTACK_HEAD} label_mode=${LABEL_MODE}"
echo "[DLG_EEGNet_batch_users] iters=${ITERS} log_every=${LOG_EVERY} max_trials=${MAX_TRIALS} start_offset=${START_OFFSET} out_dir=${OUT_DIR}"
echo "================================================================================"

exec "$PYTHON" -u -m scripts.dlg_batch_users \
  --checkpoint "$CHECKPOINT" \
  --dataset "$DATASET" \
  --data_dir "$DATA_DIR" \
  --model "$MODEL" \
  --normalize "$NORMALIZE" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --split "$SPLIT" \
  --selection "$SELECTION" \
  --attack_head "$ATTACK_HEAD" \
  --label_mode "$LABEL_MODE" \
  --iters "$ITERS" \
  --lr "$LR" \
  --optimizer "$OPTIMIZER" \
  --log_every "$LOG_EVERY" \
  --topk "$TOPK" \
  --user_hidden_dim "$USER_HIDDEN_DIM" \
  --user_dropout "$USER_DROPOUT" \
  --plot_channel "$PLOT_CHANNEL" \
  --sfreq "$SFREQ" \
  --max_trials "$MAX_TRIALS" \
  --start_offset "$START_OFFSET" \
  --out_dir "$OUT_DIR" \
  "${EXTRA_ARGS[@]}"

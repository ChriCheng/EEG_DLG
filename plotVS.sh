#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/ubuntu/miniconda3/envs/PP310/bin/python}

if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "Usage: $0 <result_dir_or_csv_A> <result_dir_or_csv_B> [out_dir] [label_A] [label_B]" >&2
  exit 2
fi

resolve_csv() {
  local input="$1"
  if [[ -d "$input" ]]; then
    input="${input%/}/batch_metrics.csv"
  fi
  if [[ ! -f "$input" ]]; then
    echo "Cannot find CSV: $input" >&2
    exit 1
  fi
  printf '%s\n' "$input"
}

A_INPUT="$1"
B_INPUT="$2"
A_CSV="$(resolve_csv "$A_INPUT")"
B_CSV="$(resolve_csv "$B_INPUT")"
A_LABEL="${4:-None noise}"
B_LABEL="${5:-Add noise}"

if [[ $# -ge 3 ]]; then
  OUT_DIR="$3"
else
  OUT_DIR="checkpoint/dlg_attack_batch/compare_$(basename "$(dirname "$A_CSV")")_VS_$(basename "$(dirname "$B_CSV")")"
fi

echo
echo "================================================================================"
echo "[plotVS] A       : $A_CSV"
echo "[plotVS] B       : $B_CSV"
echo "[plotVS] label A : $A_LABEL"
echo "[plotVS] label B : $B_LABEL"
echo "[plotVS] out_dir : $OUT_DIR"
echo "================================================================================"

exec "$PYTHON" -m scripts.plot_dlg_noise_comparison \
  --out_dir "$OUT_DIR" \
  --series "$A_LABEL=$A_CSV" "$B_LABEL=$B_CSV"


# ./plotVS.sh   checkpoint/dlg_attack_batch/P300_EEGNet_user_idlg_seed0_train2_eval4_all_eps0_sens1p0   checkpoint/dlg_attack_batch/P300_EEGNet_user_idlg_seed0_train2_eval4_all_eps10_sens1p0   checkpoint/dlg_attack_batch/compare_user_eval4_eps0_vs_eps10_v2None noise\ ep10
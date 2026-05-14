#!/usr/bin/env bash
set -euo pipefail

python_bin="/home/ubuntu/miniconda3/envs/PP310/bin/python"
base_dir="checkpoint/dlg_attack_batch"

# ==============================================================================
# plotVS 参数填写区
#
# 默认会从 checkpoint/dlg_attack_batch/ 下面找数据。
# 所以 none_noise_path / noise_path 只需要填最后一级目录名即可，例如：
#   none_noise_path="P300_EEGNet_task_idlg_seed0_train2_eval4_balanced_per_user_eps0_sens1p0"
#   noise_path="P300_EEGNet_task_idlg_seed0_train2_eval4_balanced_per_user_eps10_sens1p0"
#
# 也可以填完整目录、完整 batch_metrics.csv 路径。
#
# 1. none_noise_path  : 无噪声结果目录名，或 batch_metrics.csv 路径；通常填 eps0
# 2. noise_path       : 加噪声结果目录名，或 batch_metrics.csv 路径；通常填 eps10/eps0p1 等
# 3. out_name         : 输出目录名；留空则自动生成 compare_none_noise_vs_noise
# 4. none_noise_label : 图例中无噪声组显示的名字
# 5. noise_label      : 图例中加噪声组显示的名字
# ==============================================================================
none_noise_path="P300_EEGNet_task_idlg_seed0_train2_eval4_balanced_per_user_eps0_sens1p0"
noise_path="P300_EEGNet_task_idlg_seed0_train2_eval4_balanced_per_user_eps10_sens1p0"
out_name="compare_user_eval4_eps0_vs_eps10"
none_noise_label="None noise"
noise_label="Add noise"

resolve_csv() {
  local input="$1"
  if [[ "$input" != */* && ! -e "$input" ]]; then
    input="${base_dir%/}/$input"
  fi
  if [[ -d "$input" ]]; then
    input="${input%/}/batch_metrics.csv"
  fi
  if [[ ! -f "$input" ]]; then
    echo "Cannot find CSV: $input" >&2
    exit 1
  fi
  printf '%s\n' "$input"
}

resolve_out_dir() {
  local input="$1"
  local a_csv="$2"
  local b_csv="$3"

  if [[ -z "$input" ]]; then
    printf '%s\n' "${base_dir%/}/compare_$(basename "$(dirname "$a_csv")")_vs_$(basename "$(dirname "$b_csv")")"
  elif [[ "$input" != */* ]]; then
    printf '%s\n' "${base_dir%/}/$input"
  else
    printf '%s\n' "$input"
  fi
}

if [[ $# -gt 5 ]]; then
  echo "usage: $0 [none_noise_dir_or_csv] [noise_dir_or_csv] [out_dir_or_name] [none_noise_label] [noise_label]" >&2
  exit 2
fi

if [[ $# -ge 1 ]]; then none_noise_path="$1"; fi
if [[ $# -ge 2 ]]; then noise_path="$2"; fi
if [[ $# -ge 3 ]]; then out_name="$3"; fi
if [[ $# -ge 4 ]]; then none_noise_label="$4"; fi
if [[ $# -ge 5 ]]; then noise_label="$5"; fi

if [[ -z "$none_noise_path" || -z "$noise_path" ]]; then
  echo "Please fill none_noise_path and noise_path in plotVS.sh, or pass them as arguments." >&2
  echo "default data root: ${base_dir%/}/" >&2
  echo "usage: $0 [none_noise_dir_or_csv] [noise_dir_or_csv] [out_dir_or_name] [none_noise_label] [noise_label]" >&2
  exit 2
fi

none_noise_input="$none_noise_path"
noise_input="$noise_path"
none_noise_csv="$(resolve_csv "$none_noise_input")"
noise_csv="$(resolve_csv "$noise_input")"
out_dir="$(resolve_out_dir "$out_name" "$none_noise_csv" "$noise_csv")"

echo
echo "================================================================================"
echo "[plotVS] base    : ${base_dir%/}/"
echo "[plotVS] none noise path : $none_noise_csv"
echo "[plotVS] noise path      : $noise_csv"
echo "[plotVS] none noise label: $none_noise_label"
echo "[plotVS] noise label     : $noise_label"
echo "[plotVS] out_dir : $out_dir"
echo "================================================================================"

exec "$python_bin" -m scripts.plot_dlg_noise_comparison \
  --out_dir "$out_dir" \
  --series "$none_noise_label=$none_noise_csv" "$noise_label=$noise_csv"


# Example:
#   ./plotVS.sh \
#     P300_EEGNet_task_idlg_seed0_train2_eval4_balanced_per_user_eps0_sens1p0 \
#     P300_EEGNet_task_idlg_seed0_train2_eval4_balanced_per_user_eps10_sens1p0 \
#     compare_user_eval4_eps0_vs_eps10 \
#     "None noise" \
#     "Add noise"

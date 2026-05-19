#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# MI2_onlyDLG 参数填写区
#
# MI2_DLG 纯 DLG 单次波形复原。
# 不加载训练后 checkpoint；用 seed 固定的随机初始化模型表示训练一轮时双方共享的模型参数。
#
# 默认从人工 session 3 中抽取样本，便于只测试加入 trial-level Laplace
# 噪声后的单次 DLG 波形还原效果，不做 batch 级别 UIA 对比。
# ==============================================================================
python_bin="/home/ubuntu/miniconda3/envs/PP310/bin/python"

dataset="MI2_DLG"
model="EEGNet"
seed="0"
train_session=""
eval_session="3"
split="all"
indices="221"
batch_size="1"

attack_head="task"
label_mode="idlg"
iters="30"
lr="1.0"
optimizer="lbfgs"
lbfgs_max_iter="20"
lbfgs_line_search="strong_wolfe"
grad_loss_scale="1"
log_every="3"
topk="3"

# P300_DLG_EEGNet.sh 同款 trial-level Laplace 噪声机制。
# epsilon=0 表示不加噪；epsilon>0 时噪声 scale=sensitivity/epsilon。
# 支持临时覆盖，例如：
#   epsilon=10 trial_laplace_sensitivity=1.0 ./MI2_onlyDLG.sh
epsilon="${epsilon:-50}"
trial_laplace_sensitivity="${trial_laplace_sensitivity:-1.0}"

dummy_init_scale="auto"

plot_channel="1"
sfreq="128"
# 波形图手动时间窗（ms）。留空表示完整 0~约2000ms；例如 0~1000ms：
#   plot_xmin_ms=0 plot_xmax_ms=1000 ./MI2_onlyDLG.sh
plot_xmin_ms="${plot_xmin_ms:-}"
plot_xmax_ms="${plot_xmax_ms:-500}"
waveform_grid="0"
waveform_font_size="20"

normalize="channel"
euclidean_align="1"
device="cpu"
user_hidden_dim="256"
user_dropout="0.5"
out_dir=""

data_dir="data/${dataset}"
case "${device,,}" in
  gpu) device="cuda" ;;
  cuda) device="cuda" ;;
  cpu) device="cpu" ;;
  auto) device="auto" ;;
  *)
    echo "[MI2_onlyDLG] invalid device=$device; use auto, cpu, cuda, or gpu" >&2
    exit 2
    ;;
esac

eval_tag="${eval_session:-all}"
indices_tag="${indices:-random}"
indices_tag="${indices_tag//,/p}"
indices_tag="${indices_tag// /}"
epsilon_tag="${epsilon//./p}"
epsilon_tag="${epsilon_tag//-/m}"
epsilon_tag="${epsilon_tag//+/p}"
sensitivity_tag="${trial_laplace_sensitivity//./p}"
sensitivity_tag="${sensitivity_tag//-/m}"
sensitivity_tag="${sensitivity_tag//+/p}"
if [[ "$epsilon" == "0" || "$epsilon" == "0.0" || -z "$epsilon" ]]; then
  dp_tag="eps0_no_noise"
else
  dp_tag="eps${epsilon_tag}_sens${sensitivity_tag}"
fi

auto_out_dir="0"
if [[ -z "$out_dir" ]]; then
  auto_out_dir="1"
  out_dir="checkpoint/dlg_attack/${dataset}_onlyDLG_${model}_${attack_head}_${label_mode}_seed${seed}_split${split}_eval${eval_tag}_batch${batch_size}_trial${indices_tag}_${dp_tag}"
fi

extra_args=(--random_init_model)
if [[ "$euclidean_align" == "1" || "$euclidean_align" == "true" ]]; then
  extra_args+=(--euclidean_align)
fi
if [[ -n "$train_session" ]]; then
  extra_args+=(--train_session_internal "$((train_session - 1))")
fi
if [[ -n "$eval_session" ]]; then
  extra_args+=(--eval_session_original "$eval_session")
fi
if [[ -n "$indices" ]]; then
  extra_args+=(--indices "$indices")
fi
if [[ "$auto_out_dir" == "1" ]]; then
  extra_args+=(--append_indices_to_out_dir)
fi
if [[ "$epsilon" != "0" && "$epsilon" != "0.0" && -n "$epsilon" ]]; then
  extra_args+=(--trial_laplace_epsilon "$epsilon")
  extra_args+=(--trial_laplace_sensitivity "$trial_laplace_sensitivity")
fi
if [[ "$waveform_grid" == "0" || "$waveform_grid" == "false" || "$waveform_grid" == "False" || "$waveform_grid" == "off" || "$waveform_grid" == "no" ]]; then
  extra_args+=(--no-waveform_grid)
fi
if [[ -n "$plot_xmin_ms" ]]; then
  extra_args+=(--plot_xmin_ms "$plot_xmin_ms")
fi
if [[ -n "$plot_xmax_ms" ]]; then
  extra_args+=(--plot_xmax_ms "$plot_xmax_ms")
fi

echo
echo "================================================================================"
echo "[MI2_onlyDLG] dataset=$dataset model=$model random_init_model=1 seed=$seed"
echo "[MI2_onlyDLG] split=$split eval_session=${eval_session:-all} indices=${indices:-random} batch_size=$batch_size"
echo "[MI2_onlyDLG] attack_head=$attack_head label_mode=$label_mode iters=$iters lr=$lr optimizer=$optimizer"
echo "[MI2_onlyDLG] lbfgs_max_iter=$lbfgs_max_iter lbfgs_line_search=$lbfgs_line_search"
echo "[MI2_onlyDLG] epsilon=$epsilon sensitivity=$trial_laplace_sensitivity"
echo "[MI2_onlyDLG] grad_loss_scale=$grad_loss_scale"
echo "[MI2_onlyDLG] normalize=$normalize euclidean_align=$euclidean_align dummy_init_scale=$dummy_init_scale device=$device"
echo "[MI2_onlyDLG] plot_channel=$plot_channel plot_xmin_ms=${plot_xmin_ms:-auto} plot_xmax_ms=${plot_xmax_ms:-auto}"
echo "[MI2_onlyDLG] out_dir=$out_dir"
echo "================================================================================"

exec "$python_bin" -u -m scripts.dlg_attack \
  --dataset "$dataset" \
  --data_dir "$data_dir" \
  --model "$model" \
  --normalize "$normalize" \
  --device "$device" \
  --seed "$seed" \
  --split "$split" \
  --batch_size "$batch_size" \
  --attack_head "$attack_head" \
  --label_mode "$label_mode" \
  --iters "$iters" \
  --lr "$lr" \
  --optimizer "$optimizer" \
  --lbfgs_max_iter "$lbfgs_max_iter" \
  --lbfgs_line_search "$lbfgs_line_search" \
  --grad_loss_scale "$grad_loss_scale" \
  --log_every "$log_every" \
  --topk "$topk" \
  --dummy_init_scale "$dummy_init_scale" \
  --plot_channel "$plot_channel" \
  --sfreq "$sfreq" \
  --waveform_font_size "$waveform_font_size" \
  --user_hidden_dim "$user_hidden_dim" \
  --user_dropout "$user_dropout" \
  --out_dir "$out_dir" \
  "${extra_args[@]}"

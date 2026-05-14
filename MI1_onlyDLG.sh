#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# MI1_onlyDLG 参数填写区
#
# MI1 纯 DLG 单次波形复原。
# 不加载训练后 checkpoint；用 seed 固定的随机初始化模型表示训练一轮时双方共享的模型参数。
# indices 留空表示随机选样本；iDLG 只支持 batch_size=1。
# ==============================================================================
python_bin="/home/ubuntu/miniconda3/envs/PP310/bin/python"

dataset="MI1"
model="EEGNet"
seed="0"
train_session="1"
eval_session=""
split="test"
indices=""
batch_size="1"

attack_head="task"
label_mode="idlg"
iters="50"
lr="1.0"
optimizer="lbfgs"
lbfgs_max_iter="20"
lbfgs_line_search="strong_wolfe"
grad_loss_scale="1"
log_every="5"

dummy_init_scale="auto"

plot_channel="-1"
sfreq="128"
waveform_grid="0"
waveform_font_size="16"

normalize="trial"
euclidean_align="0"
device="cuda"
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
    echo "[MI1_onlyDLG] invalid device=$device; use auto, cpu, cuda, or gpu" >&2
    exit 2
    ;;
esac

eval_tag="${eval_session:-all}"
indices_tag="${indices:-random}"
indices_tag="${indices_tag//,/p}"
indices_tag="${indices_tag// /}"

auto_out_dir="0"
if [[ -z "$out_dir" ]]; then
  auto_out_dir="1"
  out_dir="checkpoint/dlg_attack/${dataset}_onlyDLG_${model}_${attack_head}_${label_mode}_seed${seed}_train${train_session}_split${split}_eval${eval_tag}_batch${batch_size}_trial${indices_tag}"
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
if [[ "$waveform_grid" == "0" || "$waveform_grid" == "false" || "$waveform_grid" == "False" || "$waveform_grid" == "off" || "$waveform_grid" == "no" ]]; then
  extra_args+=(--no-waveform_grid)
fi

echo
echo "================================================================================"
echo "[MI1_onlyDLG] dataset=$dataset model=$model random_init_model=1 seed=$seed"
echo "[MI1_onlyDLG] split=$split eval_session=${eval_session:-all} indices=${indices:-random} batch_size=$batch_size"
echo "[MI1_onlyDLG] attack_head=$attack_head label_mode=$label_mode iters=$iters lr=$lr optimizer=$optimizer"
echo "[MI1_onlyDLG] lbfgs_max_iter=$lbfgs_max_iter lbfgs_line_search=$lbfgs_line_search"
echo "[MI1_onlyDLG] grad_loss_scale=$grad_loss_scale"
echo "[MI1_onlyDLG] normalize=$normalize euclidean_align=$euclidean_align dummy_init_scale=$dummy_init_scale device=$device"
echo "[MI1_onlyDLG] out_dir=$out_dir"
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
  --topk "3" \
  --dummy_init_scale "$dummy_init_scale" \
  --plot_channel "$plot_channel" \
  --sfreq "$sfreq" \
  --waveform_font_size "$waveform_font_size" \
  --user_hidden_dim "$user_hidden_dim" \
  --user_dropout "$user_dropout" \
  --out_dir "$out_dir" \
  "${extra_args[@]}"

#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# DLG_EEGNet 参数填写区
#
# 单次 DLG 攻击。checkpoint 和 out_dir 留空时会按下面参数自动生成。
# eval_session 留空表示不限制 session；indices 留空表示随机选样本。
# ==============================================================================
python_bin="/home/ubuntu/miniconda3/envs/PP310/bin/python"

dataset="P300"
model="EEGNet"
seed="0"
train_session=""
eval_session=""
split="test"
indices=""
batch_size="1"

attack_head="task"
label_mode="idlg"
iters="30"
lr="1.0"
optimizer="lbfgs"
log_every="2"
topk="3"

epsilon="0"
trial_laplace_sensitivity="1.0"

plot_channel="-1"
sfreq="128"
waveform_grid="0"
waveform_font_size="16"

normalize="channel"
euclidean_align="1"
device="auto"
user_hidden_dim="256"
user_dropout="0.5"
checkpoint=""
out_dir=""

data_dir="data/${dataset}"
if [[ -z "$train_session" ]]; then
  if [[ -n "$eval_session" ]]; then
    train_session="2"
  else
    train_session="3"
  fi
fi
if [[ -n "$eval_session" ]]; then
  run_name="p300_eegnet_channel_leave_session_${eval_session}"
else
  run_name="p300_eegnet_channel"
fi
fold_name="seed_${seed}_train_session_${train_session}${eval_session:+_holdout_session_${eval_session}}"
if [[ -z "$checkpoint" ]]; then
  checkpoint="checkpoint/checkpoints_2stage_${model}/${run_name}/${fold_name}/best_user_by_acc.pth"
fi

eval_tag="${eval_session:-4}"
indices_tag="${indices:-random}"
indices_tag="${indices_tag//,/p}"
indices_tag="${indices_tag// /}"
epsilon_tag="${epsilon//./p}"
epsilon_tag="${epsilon_tag//-/m}"
epsilon_tag="${epsilon_tag//+/p}"
sensitivity_tag="${trial_laplace_sensitivity//./p}"
sensitivity_tag="${sensitivity_tag//-/m}"
sensitivity_tag="${sensitivity_tag//+/p}"
dp_tag="eps${epsilon_tag}_sens${sensitivity_tag}"

auto_out_dir="0"
if [[ -z "$out_dir" ]]; then
  auto_out_dir="1"
  out_dir="checkpoint/dlg_attack/${dataset}_${model}_${attack_head}_${label_mode}_seed${seed}_train${train_session}_split${split}_eval${eval_tag}_batch${batch_size}_trial${indices_tag}_${dp_tag}"
fi

extra_args=()
if [[ "$euclidean_align" == "1" || "$euclidean_align" == "true" ]]; then
  extra_args+=(--euclidean_align)
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

echo
echo "================================================================================"
echo "[DLG_EEGNet] dataset=$dataset model=$model checkpoint=$checkpoint"
echo "[DLG_EEGNet] split=$split eval_session=${eval_session:-all} indices=${indices:-random} batch_size=$batch_size"
echo "[DLG_EEGNet] attack_head=$attack_head label_mode=$label_mode iters=$iters lr=$lr optimizer=$optimizer"
echo "[DLG_EEGNet] epsilon=$epsilon sensitivity=$trial_laplace_sensitivity"
echo "[DLG_EEGNet] plot_channel=$plot_channel waveform_grid=$waveform_grid waveform_font_size=$waveform_font_size"
echo "[DLG_EEGNet] out_dir=$out_dir"
echo "================================================================================"

exec "$python_bin" -u -m scripts.dlg_attack \
  --checkpoint "$checkpoint" \
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
  --log_every "$log_every" \
  --topk "$topk" \
  --plot_channel "$plot_channel" \
  --sfreq "$sfreq" \
  --waveform_font_size "$waveform_font_size" \
  --user_hidden_dim "$user_hidden_dim" \
  --user_dropout "$user_dropout" \
  --out_dir "$out_dir" \
  "${extra_args[@]}"

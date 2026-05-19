#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# DLG_EEGNet_batch_users 参数填写区
#
# 批量 DLG 攻击。checkpoint 和 out_dir 留空时会按下面参数自动生成。
# none-noise 通常设 epsilon="0"，加噪声通常设 epsilon="10" 或其它值。
# ==============================================================================
python_bin="/home/ubuntu/miniconda3/envs/PP310/bin/python"

dataset="P300"
model="EEGNet"
seed="0"
train_session=""
eval_session="4"
split="test"
selection="balanced_per_user"
max_trials="200"
start_offset="0"

attack_head="task"
label_mode="idlg"
iters="33"
lr="1.0"
optimizer="lbfgs"
log_every="3"
topk="3"

epsilon="0"
trial_laplace_sensitivity="1.0"

skip_figures="1"
keep_trial_artifacts="0"
plot_channel="-1"
sfreq="128"
plot_xmin_ms="${plot_xmin_ms:-}"
plot_xmax_ms="${plot_xmax_ms:-}"
waveform_grid="1"
waveform_font_size="10"

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

epsilon_tag="${epsilon//./p}"
epsilon_tag="${epsilon_tag//-/m}"
epsilon_tag="${epsilon_tag//+/p}"
sensitivity_tag="${trial_laplace_sensitivity//./p}"
sensitivity_tag="${sensitivity_tag//-/m}"
sensitivity_tag="${sensitivity_tag//+/p}"
dp_tag="eps${epsilon_tag}_sens${sensitivity_tag}"
if [[ -z "$out_dir" ]]; then
  out_dir="checkpoint/dlg_attack_batch/${dataset}_${model}_${attack_head}_${label_mode}_seed${seed}_train${train_session}_eval${eval_session:-all}_${selection}_${dp_tag}"
fi

extra_args=()
if [[ "$euclidean_align" == "1" || "$euclidean_align" == "true" ]]; then
  extra_args+=(--euclidean_align)
fi
if [[ -n "$eval_session" ]]; then
  extra_args+=(--eval_session_original "$eval_session")
fi
if [[ "$skip_figures" == "1" || "$skip_figures" == "true" ]]; then
  extra_args+=(--skip_figures)
fi
if [[ "$keep_trial_artifacts" == "1" || "$keep_trial_artifacts" == "true" ]]; then
  extra_args+=(--keep_trial_artifacts)
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
echo "[DLG_EEGNet_batch_users] dataset=$dataset model=$model checkpoint=$checkpoint"
echo "[DLG_EEGNet_batch_users] split=$split eval_session=${eval_session:-all} selection=$selection max_trials=$max_trials start_offset=$start_offset"
echo "[DLG_EEGNet_batch_users] attack_head=$attack_head label_mode=$label_mode iters=$iters lr=$lr optimizer=$optimizer"
echo "[DLG_EEGNet_batch_users] epsilon=$epsilon sensitivity=$trial_laplace_sensitivity"
echo "[DLG_EEGNet_batch_users] skip_figures=$skip_figures keep_trial_artifacts=$keep_trial_artifacts"
echo "[DLG_EEGNet_batch_users] plot_channel=$plot_channel plot_xmin_ms=${plot_xmin_ms:-auto} plot_xmax_ms=${plot_xmax_ms:-auto}"
echo "[DLG_EEGNet_batch_users] out_dir=$out_dir"
echo "================================================================================"

exec "$python_bin" -u -m scripts.dlg_batch_users \
  --checkpoint "$checkpoint" \
  --dataset "$dataset" \
  --data_dir "$data_dir" \
  --model "$model" \
  --normalize "$normalize" \
  --device "$device" \
  --seed "$seed" \
  --split "$split" \
  --selection "$selection" \
  --attack_head "$attack_head" \
  --label_mode "$label_mode" \
  --iters "$iters" \
  --lr "$lr" \
  --optimizer "$optimizer" \
  --log_every "$log_every" \
  --topk "$topk" \
  --user_hidden_dim "$user_hidden_dim" \
  --user_dropout "$user_dropout" \
  --plot_channel "$plot_channel" \
  --sfreq "$sfreq" \
  --waveform_font_size "$waveform_font_size" \
  --max_trials "$max_trials" \
  --start_offset "$start_offset" \
  --out_dir "$out_dir" \
  "${extra_args[@]}"

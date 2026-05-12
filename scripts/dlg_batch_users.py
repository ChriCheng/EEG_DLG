from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import matplotlib
import numpy as np
import torch

from scripts.dlg_attack import run_dlg, save_text
from scripts.train import build_dataset, build_loso_split

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def mean(rows: List[Dict], key: str) -> float:
    vals = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def select_one_trial_per_user(
    ds,
    split: str,
    train_session_internal: int | None,
    seed: int,
) -> List[int]:
    if split == "all":
        pool = list(range(len(ds)))
    else:
        if train_session_internal is None:
            raise ValueError("--train_session_internal is required when --split is train or test")
        train_idx, test_idx = build_loso_split(ds, train_session_internal)
        pool = train_idx if split == "train" else test_idx

    by_user: Dict[int, List[int]] = {u: [] for u in range(ds.n_users)}
    for idx in pool:
        user = int(ds.y_user[idx].item())
        by_user[user].append(idx)

    rng = random.Random(seed)
    selected = []
    for user in range(ds.n_users):
        candidates = by_user[user]
        if not candidates:
            raise ValueError(f"No candidate trials for user={user} in split={split}")
        selected.append(rng.choice(candidates))
    return selected


def flatten_result(summary: Dict, elapsed_sec: float) -> Dict:
    topk = summary["topk"]
    topk_key = f"user_top{topk}_acc"
    recon_metrics = summary["identity_metrics"]["recon"]
    real_metrics = summary["identity_metrics"]["real"]
    noise_metrics = summary["identity_metrics"]["noise"]
    label_info = summary["label_info"]
    attack_label = int(label_info["attack_labels"][0])
    inferred = label_info.get("idlg_inferred_labels", label_info.get("optimized_labels", [None]))[0]
    inferred = None if inferred is None else int(inferred)
    label_correct = None if inferred is None else int(inferred == attack_label)
    recon_detail = summary["identity_details"]["recon"][0]
    return {
        "user": int(summary["y_user"][0]),
        "trial_index": int(summary["batch_indices"][0]),
        "task_label": int(summary["y_task"][0]),
        "attack_label": attack_label,
        "inferred_label": inferred,
        "label_correct": label_correct,
        "mse": float(summary["final_reconstruction"]["mse"]),
        "corr": float(summary["final_reconstruction"]["corr"]),
        "final_grad_loss": float(summary["history"][-1]["grad_loss"]) if summary["history"] else None,
        "recon_user_top1_acc": float(recon_metrics["user_top1_acc"]),
        f"recon_user_top{topk}_acc": float(recon_metrics[topk_key]),
        "recon_true_user_conf": float(recon_metrics["true_user_conf"]),
        "recon_task_acc": float(recon_metrics["task_acc"]),
        "real_user_top1_acc": float(real_metrics["user_top1_acc"]),
        f"real_user_top{topk}_acc": float(real_metrics[topk_key]),
        "real_true_user_conf": float(real_metrics["true_user_conf"]),
        "noise_user_top1_acc": float(noise_metrics["user_top1_acc"]),
        f"noise_user_top{topk}_acc": float(noise_metrics[topk_key]),
        "noise_true_user_conf": float(noise_metrics["true_user_conf"]),
        "recon_topk_users": recon_detail["topk_users"],
        "recon_topk_probs": recon_detail["topk_probs"],
        "elapsed_sec": float(elapsed_sec),
    }


def format_csv(rows: List[Dict], topk: int) -> str:
    fields = [
        "user",
        "trial_index",
        "task_label",
        "attack_label",
        "inferred_label",
        "label_correct",
        "mse",
        "corr",
        "final_grad_loss",
        "recon_user_top1_acc",
        f"recon_user_top{topk}_acc",
        "recon_true_user_conf",
        "recon_task_acc",
        "real_user_top1_acc",
        f"real_user_top{topk}_acc",
        "real_true_user_conf",
        "noise_user_top1_acc",
        f"noise_user_top{topk}_acc",
        "noise_true_user_conf",
        "recon_topk_users",
        "recon_topk_probs",
        "elapsed_sec",
    ]
    lines = [",".join(fields)]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            elif isinstance(value, list):
                vals.append("|".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in value))
            elif value is None:
                vals.append("")
            else:
                vals.append(str(value))
        lines.append(",".join(vals))
    return "\n".join(lines) + "\n"


def format_report(rows: List[Dict], args: argparse.Namespace, selected_indices: List[int]) -> str:
    topk = args.topk
    topk_key = f"recon_user_top{topk}_acc"
    label_rows = [row for row in rows if row["label_correct"] is not None]
    label_acc = mean(label_rows, "label_correct") if label_rows else float("nan")
    lines = [
        "DLG EEG Batch User Summary",
        "=" * 80,
        f"checkpoint       : {args.checkpoint}",
        f"dataset          : {args.dataset}",
        f"model            : {args.model}",
        f"split            : {args.split}",
        f"train_session    : {args.train_session_internal}",
        f"attack_head      : {args.attack_head}",
        f"label_mode       : {args.label_mode}",
        f"iters/log_every  : {args.iters}/{args.log_every}",
        f"selected_indices : {selected_indices}",
        "",
        "Aggregate",
        "-" * 80,
        f"n_users                 : {len(rows)}",
        f"mean MSE                : {mean(rows, 'mse'):.4f}",
        f"mean Corr               : {mean(rows, 'corr'):.4f}",
        f"iDLG label acc          : {format_pct(label_acc)}",
        f"Recon User Top-1        : {format_pct(mean(rows, 'recon_user_top1_acc'))}",
        f"Recon User Top-{topk}        : {format_pct(mean(rows, topk_key))}",
        f"Recon True-user Conf    : {mean(rows, 'recon_true_user_conf'):.4f}",
        f"Recon Task Acc          : {format_pct(mean(rows, 'recon_task_acc'))}",
        f"Real User Top-1         : {format_pct(mean(rows, 'real_user_top1_acc'))}",
        f"Noise User Top-1        : {format_pct(mean(rows, 'noise_user_top1_acc'))}",
        f"mean elapsed/sample sec : {mean(rows, 'elapsed_sec'):.4f}",
        "",
        "Per User",
        "-" * 80,
        f"{'user':>4} {'trial':>8} {'task':>4} {'label':>7} {'mse':>8} {'corr':>8} {'Top1':>7} {('Top' + str(topk)):>7} {'conf':>8} topk_users",
    ]
    for row in rows:
        topk_users = "|".join(str(v) for v in row["recon_topk_users"])
        lines.append(
            f"{row['user']:>4} {row['trial_index']:>8} {row['task_label']:>4} "
            f"{str(row['label_correct']):>7} {row['mse']:>8.4f} {row['corr']:>8.4f} "
            f"{format_pct(row['recon_user_top1_acc']):>7} "
            f"{format_pct(row[topk_key]):>7} "
            f"{row['recon_true_user_conf']:>8.4f} {topk_users}"
        )
    return "\n".join(lines) + "\n"


def choose_channel(real_x: torch.Tensor, requested_channel: int) -> int:
    if requested_channel >= 0:
        if requested_channel >= real_x.shape[0]:
            raise ValueError(
                f"plot_channel={requested_channel} is out of range for n_channels={real_x.shape[0]}"
            )
        return requested_channel
    return int(torch.var(real_x, dim=1).argmax().item())


def plot_waveform_grid(rows: List[Dict], out_dir: Path, plot_channel: int, sfreq: float) -> str:
    cols = min(4, len(rows))
    rows_n = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.8 * cols, 3.0 * rows_n), squeeze=False)

    for ax, row in zip(axes.flat, rows):
        recon_path = out_dir / f"user_{row['user']:02d}_trial_{row['trial_index']}" / "reconstruction.pt"
        payload = torch.load(recon_path, map_location="cpu")
        real_x = payload["real_x"][0]
        recon_x = payload["recon_x"][0]
        channel = choose_channel(real_x, plot_channel)
        time_ms = np.arange(real_x.shape[1], dtype=np.float32) / float(sfreq) * 1000.0

        ax.plot(time_ms, real_x[channel].numpy(), label="target", linewidth=1.5, color="#111111")
        ax.plot(time_ms, recon_x[channel].numpy(), label="recon", linewidth=1.1, color="#d95f02", alpha=0.9)
        ax.set_title(f"user {row['user']} | ch {channel} | corr {row['corr']:.4f}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (a.u.)")
        ax.grid(alpha=0.22)

    for ax in axes.flat[len(rows):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Per-user EEG reconstruction waveform", y=0.985)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = out_dir / "batch_waveform_grid.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_quality_bars(rows: List[Dict], out_dir: Path) -> str:
    users = [row["user"] for row in rows]
    mse = [row["mse"] for row in rows]
    corr = [row["corr"] for row in rows]
    x = np.arange(len(users))

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), sharex=True)
    axes[0].bar(x, mse, color="#4c78a8")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Reconstruction error per user")
    axes[0].grid(axis="y", alpha=0.22)

    axes[1].bar(x, corr, color="#59a14f")
    axes[1].set_ylabel("Correlation")
    axes[1].set_xlabel("User")
    axes[1].set_title("Reconstruction correlation per user")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(u) for u in users])
    axes[1].grid(axis="y", alpha=0.22)

    fig.tight_layout()
    path = out_dir / "batch_quality_bars.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_leakage_bars(rows: List[Dict], out_dir: Path, topk: int) -> str:
    topk_key = f"recon_user_top{topk}_acc"
    real_topk_key = f"real_user_top{topk}_acc"
    noise_topk_key = f"noise_user_top{topk}_acc"
    labels = ["Real EEG", "Reconstructed", "Noise"]
    metrics = {
        "Top-1 Acc": [
            mean(rows, "real_user_top1_acc"),
            mean(rows, "recon_user_top1_acc"),
            mean(rows, "noise_user_top1_acc"),
        ],
        f"Top-{topk} Acc": [
            mean(rows, real_topk_key),
            mean(rows, topk_key),
            mean(rows, noise_topk_key),
        ],
        "True-user Conf": [
            mean(rows, "real_true_user_conf"),
            mean(rows, "recon_true_user_conf"),
            mean(rows, "noise_true_user_conf"),
        ],
    }

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for offset, (name, values) in enumerate(metrics.items()):
        ax.bar(x + (offset - 1) * width, values, width, label=name, color=colors[offset])
        for xi, value in zip(x + (offset - 1) * width, values):
            ax.text(xi, value + 0.015, f"{value * 100:.1f}%" if "Conf" not in name else f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Identity leakage: real vs reconstructed vs noise")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    path = out_dir / "batch_identity_leakage_bars.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one DLG/iDLG reconstruction per P300 user.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="P300")
    parser.add_argument("--data_dir", type=str, default="data/P300")
    parser.add_argument("--model", type=str, default="EEGNet", choices=["EEGNet", "LMEEGNet", "ShallowCNN"])
    parser.add_argument("--normalize", type=str, default="channel", choices=["none", "trial", "channel"])
    parser.add_argument("--euclidean_align", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"])
    parser.add_argument("--train_session_internal", type=int, default=None)
    parser.add_argument("--attack_head", type=str, default="task", choices=["task", "user"])
    parser.add_argument("--label_mode", type=str, default="idlg", choices=["idlg", "true", "dlg"])
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="lbfgs", choices=["lbfgs", "adam"])
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--user_hidden_dim", type=int, default=256)
    parser.add_argument("--user_dropout", type=float, default=0.5)
    parser.add_argument("--plot_channel", type=int, default=-1)
    parser.add_argument("--sfreq", type=float, default=128.0)
    parser.add_argument("--out_dir", type=str, default="checkpoint/dlg_attack_batch")
    args = parser.parse_args()

    ds = build_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        normalize=args.normalize,
        euclidean_align=args.euclidean_align,
    )
    train_session_internal = args.train_session_internal
    if train_session_internal is None:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        train_session_internal = ckpt.get("extra", {}).get("train_session_internal")
        args.train_session_internal = train_session_internal

    selected_indices = select_one_trial_per_user(
        ds=ds,
        split=args.split,
        train_session_internal=train_session_internal,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    summaries = []
    for user, trial_idx in enumerate(selected_indices):
        user_out_dir = out_dir / f"user_{user:02d}_trial_{trial_idx}"
        attack_args = SimpleNamespace(**vars(args))
        attack_args.indices = str(trial_idx)
        attack_args.batch_size = 1
        attack_args.out_dir = str(user_out_dir)
        print("\n" + "=" * 80)
        print(f"[Batch DLG] user={user} trial_index={trial_idx} out={user_out_dir}")
        print("=" * 80)
        start = time.time()
        summary = run_dlg(attack_args)
        elapsed = time.time() - start
        row = flatten_result(summary, elapsed_sec=elapsed)
        rows.append(row)
        summaries.append(summary)

    batch_summary = {
        "args": vars(args),
        "selected_indices": selected_indices,
        "rows": rows,
        "summaries": summaries,
    }
    save_json(batch_summary, out_dir / "batch_summary.json")
    save_text(format_csv(rows, args.topk), out_dir / "batch_metrics.csv")
    save_text(format_report(rows, args, selected_indices), out_dir / "batch_summary.txt")
    figure_paths = {
        "waveform_grid": plot_waveform_grid(rows, out_dir, args.plot_channel, args.sfreq),
        "quality_bars": plot_quality_bars(rows, out_dir),
        "identity_leakage_bars": plot_leakage_bars(rows, out_dir, args.topk),
    }
    batch_summary["figures"] = figure_paths
    save_json(batch_summary, out_dir / "batch_summary.json")
    print("\nSaved:")
    print(f"  - {out_dir / 'batch_summary.json'}")
    print(f"  - {out_dir / 'batch_metrics.csv'}")
    print(f"  - {out_dir / 'batch_summary.txt'}")
    for path in figure_paths.values():
        print(f"  - {path}")


if __name__ == "__main__":
    main()

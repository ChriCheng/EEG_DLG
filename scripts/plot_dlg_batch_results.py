from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


FLOAT_FIELDS = {
    "mse",
    "corr",
    "final_grad_loss",
    "recon_user_top1_acc",
    "recon_user_top3_acc",
    "recon_true_user_conf",
    "recon_task_acc",
    "real_user_top1_acc",
    "real_user_top3_acc",
    "real_true_user_conf",
    "noise_user_top1_acc",
    "noise_user_top3_acc",
    "noise_true_user_conf",
    "elapsed_sec",
}
INT_FIELDS = {
    "user",
    "trial_index",
    "session_internal",
    "session_original",
    "task_label",
    "attack_label",
    "inferred_label",
    "label_correct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot aggregate DLG batch metrics from one or more CSV files.")
    parser.add_argument("--metrics", nargs="+", required=True, help="Input batch_metrics.csv files.")
    parser.add_argument("--out_dir", required=True, help="Directory for figures and combined summary.")
    parser.add_argument("--title", default="DLG batch results", help="Figure title prefix.")
    return parser.parse_args()


def load_rows(paths: Iterable[str]) -> List[Dict]:
    rows: List[Dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parsed = dict(row)
                for key in INT_FIELDS:
                    parsed[key] = int(parsed[key])
                for key in FLOAT_FIELDS:
                    parsed[key] = float(parsed[key])
                parsed["source_csv"] = str(path)
                rows.append(parsed)
    rows.sort(key=lambda item: item["trial_index"])
    return rows


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    left_pad = window // 2
    right_pad = window - 1 - left_pad
    padded = np.pad(values, (left_pad, right_pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def save_quality_trends(rows: List[Dict], out_dir: Path, title: str) -> str:
    sample_ids = np.arange(1, len(rows) + 1)
    mse = np.array([row["mse"] for row in rows], dtype=np.float64)
    corr = np.array([row["corr"] for row in rows], dtype=np.float64)
    window = min(25, max(5, len(rows) // 16))

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=True)
    axes[0].plot(sample_ids, mse, linewidth=0.9, alpha=0.35, color="#4c78a8", label="Per sample")
    axes[0].plot(sample_ids, rolling_mean(mse, window), linewidth=2.0, color="#1f4e79", label=f"Rolling mean ({window})")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Reconstruction error across processed samples")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(sample_ids, corr, linewidth=0.9, alpha=0.35, color="#59a14f", label="Per sample")
    axes[1].plot(sample_ids, rolling_mean(corr, window), linewidth=2.0, color="#2f6f3e", label=f"Rolling mean ({window})")
    axes[1].set_xlabel("Processed sample index")
    axes[1].set_ylabel("Correlation")
    axes[1].set_title("Reconstruction correlation across processed samples")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, loc="lower right")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out_dir / "dlg_400_quality_trends.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def save_leakage_summary(rows: List[Dict], out_dir: Path, title: str) -> str:
    labels = ["Real EEG", "Reconstructed", "Noise"]
    metrics = {
        "Top-1 Acc": [
            mean(row["real_user_top1_acc"] for row in rows),
            mean(row["recon_user_top1_acc"] for row in rows),
            mean(row["noise_user_top1_acc"] for row in rows),
        ],
        "Top-3 Acc": [
            mean(row["real_user_top3_acc"] for row in rows),
            mean(row["recon_user_top3_acc"] for row in rows),
            mean(row["noise_user_top3_acc"] for row in rows),
        ],
        "True-user Conf": [
            mean(row["real_true_user_conf"] for row in rows),
            mean(row["recon_true_user_conf"] for row in rows),
            mean(row["noise_true_user_conf"] for row in rows),
        ],
    }

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for offset, (name, values) in enumerate(metrics.items()):
        positions = x + (offset - 1) * width
        ax.bar(positions, values, width, label=name, color=colors[offset])
        for xpos, value in zip(positions, values):
            label = f"{value * 100:.1f}%" if "Conf" not in name else f"{value:.3f}"
            ax.text(xpos, value + 0.018, label, ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Identity leakage comparison")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out_dir / "dlg_400_identity_leakage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def save_confidence_distribution(rows: List[Dict], out_dir: Path, title: str) -> str:
    reconstructed = [row["recon_true_user_conf"] for row in rows]
    real = [row["real_true_user_conf"] for row in rows]
    noise = [row["noise_true_user_conf"] for row in rows]

    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    bins = np.linspace(0.0, 1.0, 26)
    ax.hist(real, bins=bins, alpha=0.55, label="Real EEG", color="#4c78a8")
    ax.hist(reconstructed, bins=bins, alpha=0.55, label="Reconstructed", color="#f58518")
    ax.hist(noise, bins=bins, alpha=0.55, label="Noise", color="#54a24b")
    ax.set_xlabel("True-user confidence")
    ax.set_ylabel("Sample count")
    ax.set_title("Confidence distribution over processed samples")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out_dir / "dlg_400_true_user_confidence_hist.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def summarize(rows: List[Dict], figures: Dict[str, str]) -> Dict:
    users = Counter(row["user"] for row in rows)
    summary = {
        "n_trials": len(rows),
        "trial_index_min": min(row["trial_index"] for row in rows),
        "trial_index_max": max(row["trial_index"] for row in rows),
        "users": dict(sorted(users.items())),
        "mean_mse": mean(row["mse"] for row in rows),
        "std_mse": pstdev(row["mse"] for row in rows),
        "mean_corr": mean(row["corr"] for row in rows),
        "median_corr": median(row["corr"] for row in rows),
        "idlg_label_acc": mean(row["label_correct"] for row in rows),
        "recon_user_top1_acc": mean(row["recon_user_top1_acc"] for row in rows),
        "recon_user_top3_acc": mean(row["recon_user_top3_acc"] for row in rows),
        "recon_true_user_conf": mean(row["recon_true_user_conf"] for row in rows),
        "recon_task_acc": mean(row["recon_task_acc"] for row in rows),
        "real_user_top1_acc": mean(row["real_user_top1_acc"] for row in rows),
        "noise_user_top1_acc": mean(row["noise_user_top1_acc"] for row in rows),
        "mean_elapsed_sec": mean(row["elapsed_sec"] for row in rows),
        "figures": figures,
    }
    return summary


def main() -> None:
    args = parse_args()
    rows = load_rows(args.metrics)
    if not rows:
        raise ValueError("No rows loaded from the provided metrics files.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    title = f"{args.title} | n={len(rows)}"

    figures = {
        "quality_trends": save_quality_trends(rows, out_dir, title),
        "identity_leakage": save_leakage_summary(rows, out_dir, title),
        "true_user_confidence_hist": save_confidence_distribution(rows, out_dir, title),
    }
    summary = summarize(rows, figures)

    summary_path = out_dir / "dlg_400_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Loaded {len(rows)} rows")
    print(f"Saved {summary_path}")
    for path in figures.values():
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

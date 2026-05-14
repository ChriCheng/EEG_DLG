from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


DEFAULT_SERIES = [
    (
        "No noise",
        "checkpoint/dlg_attack_batch/P300_EEGNet_task_idlg_seed0_train3_eval4_all/batch_metrics.csv",
    ),
    (
        "DLG_1 eps=1.0",
        "checkpoint/dlg_attack_batch/P300_EEGNet_task_idlg_seed0_train3_evalall_all/batch_metrics.csv",
    ),
    (
        "DLG_01 eps=0.1",
        "checkpoint/dlg_attack_batch/P300_EEGNet_task_idlg_seed0_train3_evalall_all_eps0p1_sens1p0/batch_metrics.csv",
    ),
]

FLOAT_FIELDS = {
    "mse",
    "corr",
    "mse_to_clean",
    "corr_to_clean",
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
    "real_task_acc",
    "real_true_task_conf",
    "noise_task_acc",
    "noise_true_task_conf",
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

COLORS = {
    "No noise": "#2f6f3e",
    "DLG_1 eps=1.0": "#4c78a8",
    "DLG_01 eps=0.1": "#d95f02",
}
FALLBACK_COLORS = ["#4c78a8", "#d95f02", "#2f6f3e", "#e45756", "#72b7b2", "#b279a2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DLG noisy runs from one or more batch metric CSV files.")
    parser.add_argument("--out_dir", default="checkpoint/dlg_attack_batch/noise_comparison_figures")
    parser.add_argument(
        "--series",
        nargs="*",
        default=[],
        help="Optional series as label=path. Defaults to no-noise, DLG_1, and DLG_01.",
    )
    return parser.parse_args()


def parse_series(items: Iterable[str]) -> List[tuple[str, str]]:
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Series must be label=path, got: {item}")
        label, path = item.split("=", 1)
        parsed.append((label.strip(), path.strip()))
    return parsed or DEFAULT_SERIES


def maybe_number(row: Dict[str, str], field: str):
    value = row.get(field, "")
    if value == "":
        return None
    if field in INT_FIELDS:
        return int(value)
    if field in FLOAT_FIELDS:
        return float(value)
    return value


def load_rows(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            row = {key: maybe_number(raw, key) for key in raw}
            rows.append(row)
    rows.sort(key=lambda item: item["trial_index"])
    return rows


def sample_count_label(series_rows: Dict[str, List[Dict]]) -> str:
    counts = sorted({len(rows) for rows in series_rows.values()})
    if len(counts) == 1:
        return f"{counts[0]} samples"
    return " / ".join(str(count) for count in counts) + " samples"


def series_color(label: str, index: int) -> str:
    return COLORS.get(label, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    left_pad = window // 2
    right_pad = window - 1 - left_pad
    padded = np.pad(values, (left_pad, right_pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def values(rows: List[Dict], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows if row.get(key) is not None], dtype=np.float64)


def save_quality_trends(series_rows: Dict[str, List[Dict]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.2), sharex=True)
    window = 15

    for idx, (label, rows) in enumerate(series_rows.items()):
        x = np.arange(1, len(rows) + 1)
        color = series_color(label, idx)
        mse = values(rows, "mse")
        corr = values(rows, "corr")
        axes[0].plot(x, mse, linewidth=0.8, alpha=0.22, color=color)
        axes[0].plot(x, rolling_mean(mse, window), linewidth=2.0, color=color, label=label)
        axes[1].plot(x, corr, linewidth=0.8, alpha=0.22, color=color)
        axes[1].plot(x, rolling_mean(corr, window), linewidth=2.0, color=color, label=label)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("MSE (log scale)")
    axes[0].set_title(f"Reconstruction error over {sample_count_label(series_rows)}")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].set_xlabel("Sample number")
    axes[1].set_ylabel("Correlation")
    axes[1].set_title(f"Reconstruction correlation over {sample_count_label(series_rows)}")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, loc="lower left")

    fig.suptitle("DLG noise comparison")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out_dir / "dlg_noise_comparison_quality_trends.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_metric_bars(series_rows: Dict[str, List[Dict]], out_dir: Path) -> Path:
    metrics = [
        ("MSE", "mse", True),
        ("Corr", "corr", False),
        ("Recon UIA 1", "recon_user_top1_acc", False),
        ("Recon UIA 3", "recon_user_top3_acc", False),
        ("Real UIA 1", "real_user_top1_acc", False),
        ("True-user Conf", "recon_true_user_conf", False),
    ]
    labels = list(series_rows)
    x = np.arange(len(metrics))
    width = 0.24

    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    for idx, label in enumerate(labels):
        rows = series_rows[label]
        bar_values = [mean(values(rows, key)) for _, key, _ in metrics]
        positions = x + (idx - (len(labels) - 1) / 2.0) * width
        ax.bar(positions, bar_values, width, label=label, color=series_color(label, idx))
        for xpos, value in zip(positions, bar_values):
            text = f"{value:.3f}" if value < 10 else f"{value:.1f}"
            ax.text(xpos, value * 1.05 if value > 0 else 0.015, text, ha="center", va="bottom", fontsize=8)

    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("Mean value (symlog scale)")
    ax.set_title(f"Mean metrics over {sample_count_label(series_rows)}")
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _, _ in metrics])
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    path = out_dir / "dlg_noise_comparison_metric_bars.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_identity_metrics(series_rows: Dict[str, List[Dict]], out_dir: Path) -> Path:
    metrics = [
        ("iDLG Label Acc", "label_correct"),
        ("Recon UIA Top-1", "recon_user_top1_acc"),
        ("Recon UIA Top-3", "recon_user_top3_acc"),
        ("Recon True-user Conf", "recon_true_user_conf"),
        ("Real-signal UIA Top-1", "real_user_top1_acc"),
        ("Real True-user Conf", "real_true_user_conf"),
    ]
    labels = list(series_rows)
    x = np.arange(len(metrics))
    width = min(0.34, 0.78 / max(len(labels), 1))

    fig, ax = plt.subplots(figsize=(14.2, 5.8))
    for idx, label in enumerate(labels):
        rows = series_rows[label]
        bar_values = [mean(values(rows, key)) for _, key in metrics]
        positions = x + (idx - (len(labels) - 1) / 2.0) * width
        color = series_color(label, idx)
        ax.bar(positions, bar_values, width, label=label, color=color, alpha=0.86)
        for xpos, value in zip(positions, bar_values):
            ax.text(xpos, value + 0.018, f"{value * 100:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Rate / confidence")
    ax.set_title("Identity leakage metrics on reconstructed signals")
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in metrics])
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path = out_dir / "dlg_identity_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_user_prediction_distribution(series_rows: Dict[str, List[Dict]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8))
    users = sorted(
        {
            int(row["user"])
            for rows in series_rows.values()
            for row in rows
            if row.get("user") is not None
        }
        | {
            int(str(row["recon_topk_users"]).split("|")[0])
            for rows in series_rows.values()
            for row in rows
            if row.get("recon_topk_users") not in (None, "")
        }
    )
    if not users:
        users = list(range(8))
    x = np.arange(len(users))
    width = min(0.18, 0.78 / max(len(series_rows) + 1, 1))
    user_to_pos = {user: pos for pos, user in enumerate(users)}
    true_counts = np.zeros(len(users), dtype=float)
    first_rows = next(iter(series_rows.values()), [])
    for row in first_rows:
        true_user = row.get("user")
        if true_user is not None:
            true_counts[user_to_pos[int(true_user)]] += 1

    true_positions = x - (len(series_rows) / 2.0) * width
    axes[0].bar(
        true_positions,
        true_counts,
        width=width,
        label="True",
        color="#6b7280",
        alpha=0.34,
        edgecolor="#374151",
        hatch="//",
    )
    for xpos, count in zip(true_positions, true_counts):
        if count:
            axes[0].text(xpos, count + 1.0, f"{int(count)}", ha="center", va="bottom", fontsize=8)

    for idx, (label, rows) in enumerate(series_rows.items()):
        pred_counts = np.zeros(len(users), dtype=float)
        for row in rows:
            top_user = str(row.get("recon_topk_users", "")).split("|")[0]
            if top_user != "":
                pred_counts[user_to_pos[int(top_user)]] += 1
        pred_positions = x + (idx + 1 - len(series_rows) / 2.0) * width
        color = series_color(label, idx)
        axes[0].bar(
            pred_positions,
            pred_counts,
            width=width,
            label=label,
            color=color,
            alpha=0.84,
        )
        for xpos, count in zip(pred_positions, pred_counts):
            if count:
                axes[0].text(xpos, count + 1.0, f"{int(count)}", ha="center", va="bottom", fontsize=8)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"user {user}" for user in users], rotation=0, ha="center")
    axes[0].set_ylabel("Sample count")
    axes[0].set_title("True vs predicted identity counts")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)

    bins = np.linspace(0.0, 1.0, 31)
    for idx, (label, rows) in enumerate(series_rows.items()):
        conf = values(rows, "recon_true_user_conf")
        color = series_color(label, idx)
        axes[1].hist(conf, bins=bins, alpha=0.45, color=color, label=label)
        axes[1].axvline(float(np.mean(conf)), color=color, linewidth=2.0)
    axes[1].set_xlabel("Recon true-user confidence")
    axes[1].set_ylabel("Sample count")
    axes[1].set_title("Confidence assigned to the true user")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    path = out_dir / "dlg_user_prediction_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_distributions(series_rows: Dict[str, List[Dict]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    for idx, (label, rows) in enumerate(series_rows.items()):
        color = series_color(label, idx)
        axes[0].hist(values(rows, "corr"), bins=np.linspace(0.0, 1.0, 31), alpha=0.45, color=color, label=label)
        mse = np.maximum(values(rows, "mse"), 1e-12)
        axes[1].hist(np.log10(mse), bins=30, alpha=0.45, color=color, label=label)

    axes[0].set_xlabel("Correlation")
    axes[0].set_ylabel("Sample count")
    axes[0].set_title("Correlation distribution")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False)

    axes[1].set_xlabel("log10(MSE)")
    axes[1].set_ylabel("Sample count")
    axes[1].set_title("MSE distribution")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    path = out_dir / "dlg_noise_comparison_distributions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def summarize(series_rows: Dict[str, List[Dict]], figures: Dict[str, str]) -> Dict:
    summary = {"figures": figures, "series": {}}
    for label, rows in series_rows.items():
        trial_indices = [row["trial_index"] for row in rows]
        summary["series"][label] = {
            "n": len(rows),
            "trial_index_min": min(trial_indices),
            "trial_index_max": max(trial_indices),
            "mean_mse": mean(values(rows, "mse")),
            "std_mse": pstdev(values(rows, "mse")),
            "median_mse": median(values(rows, "mse")),
            "mean_corr": mean(values(rows, "corr")),
            "median_corr": median(values(rows, "corr")),
            "recon_user_top1_acc": mean(values(rows, "recon_user_top1_acc")),
            "recon_user_top3_acc": mean(values(rows, "recon_user_top3_acc")),
            "recon_true_user_conf": mean(values(rows, "recon_true_user_conf")),
            "real_user_top1_acc": mean(values(rows, "real_user_top1_acc")),
            "real_user_top3_acc": mean(values(rows, "real_user_top3_acc")),
            "real_true_user_conf": mean(values(rows, "real_true_user_conf")),
            "idlg_label_acc": mean(values(rows, "label_correct")),
            "recon_task_acc": mean(values(rows, "recon_task_acc")),
            "real_task_acc": mean(values(rows, "real_task_acc")),
        }
        if any(row.get("mse_to_clean") is not None for row in rows):
            summary["series"][label]["mean_mse_to_clean"] = mean(values(rows, "mse_to_clean"))
            summary["series"][label]["mean_corr_to_clean"] = mean(values(rows, "corr_to_clean"))
    return summary


def save_summary_csv(summary: Dict, out_dir: Path) -> Path:
    fields = [
        "series",
        "n",
        "trial_index_min",
        "trial_index_max",
        "mean_mse",
        "std_mse",
        "median_mse",
        "mean_corr",
        "median_corr",
        "mean_mse_to_clean",
        "mean_corr_to_clean",
        "recon_user_top1_acc",
        "recon_user_top3_acc",
        "recon_true_user_conf",
        "real_user_top1_acc",
        "real_user_top3_acc",
        "real_true_user_conf",
        "idlg_label_acc",
        "recon_task_acc",
        "real_task_acc",
    ]
    path = out_dir / "dlg_noise_comparison_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, values_ in summary["series"].items():
            row = {"series": label}
            row.update(values_)
            writer.writerow(row)
    return path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series_rows = {}
    for label, raw_path in parse_series(args.series):
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        series_rows[label] = load_rows(path)

    figures = {
        "quality_trends": str(save_quality_trends(series_rows, out_dir)),
        "metric_bars": str(save_metric_bars(series_rows, out_dir)),
        "identity_metrics": str(save_identity_metrics(series_rows, out_dir)),
        "user_prediction_distribution": str(save_user_prediction_distribution(series_rows, out_dir)),
        "distributions": str(save_distributions(series_rows, out_dir)),
    }
    summary = summarize(series_rows, figures)
    summary_path = out_dir / "dlg_noise_comparison_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    csv_path = save_summary_csv(summary, out_dir)

    print(f"Saved {summary_path}")
    print(f"Saved {csv_path}")
    for path in figures.values():
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

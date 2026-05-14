from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.train import build_dataset, build_loso_split, build_model

matplotlib.use("Agg")
from matplotlib import pyplot as plt


class LinearEEGModel(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_task_classes: int, n_users: int):
        super().__init__()
        input_dim = n_channels * n_times
        self.task_head = nn.Linear(input_dim, n_task_classes)
        self.user_head = nn.Linear(input_dim, n_users)

    def forward(self, x: torch.Tensor, head: str = "task") -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        if head == "task":
            return self.task_head(flat)
        if head == "user":
            return self.user_head(flat)
        raise ValueError(f"Unknown head={head!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_text(text: str, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def infer_user_hidden_dim(state_dict: Dict[str, torch.Tensor], default: int) -> int:
    weight = state_dict.get("user_head.fc1.weight")
    if weight is None:
        return default
    return int(weight.shape[0])


def load_checkpoint_model(
    checkpoint_path: Path,
    ds,
    model_name: str,
    user_hidden_dim: int,
    user_dropout: float,
    device: torch.device,
) -> Tuple[nn.Module, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = strip_module_prefix(checkpoint["model_state_dict"])
    hidden_dim = infer_user_hidden_dim(state_dict, user_hidden_dim)
    model, _ = build_model(
        ds,
        user_hidden_dim=hidden_dim,
        model_name=model_name,
        user_dropout=user_dropout,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def build_random_init_model(
    ds,
    model_name: str,
    user_hidden_dim: int,
    user_dropout: float,
    device: torch.device,
) -> Tuple[nn.Module, Dict]:
    if model_name == "Linear":
        cfg = {
            "n_channels": ds.n_channels,
            "n_times": ds.n_times,
            "n_task_classes": ds.n_task_classes,
            "n_users": ds.n_users,
        }
        model = LinearEEGModel(
            n_channels=ds.n_channels,
            n_times=ds.n_times,
            n_task_classes=ds.n_task_classes,
            n_users=ds.n_users,
        )
    else:
        model, cfg = build_model(
            ds,
            user_hidden_dim=user_hidden_dim,
            model_name=model_name,
            user_dropout=user_dropout,
        )
    model.to(device)
    model.eval()
    return model, {
        "epoch": 0,
        "model_config": cfg,
        "extra": {
            "stage": "random_init",
            "dataset_type": getattr(ds, "dataset_name", None),
            "model_name": model_name,
        },
    }


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum(-target_probs * F.log_softmax(logits, dim=1), dim=1))


def named_trainable_parameters(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def head_logits(model: nn.Module, x: torch.Tensor, head: str) -> torch.Tensor:
    out = model(x, head=head)
    if isinstance(out, tuple):
        raise ValueError(f"Expected logits for head={head!r}, got tuple output")
    return out


def gradients_from_batch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    head: str,
    create_graph: bool,
    named_params: List[Tuple[str, nn.Parameter]],
) -> Tuple[torch.Tensor, List[torch.Tensor | None]]:
    logits = head_logits(model, x, head=head)
    loss = F.cross_entropy(logits, y)
    params = [p for _, p in named_params]
    grads = torch.autograd.grad(loss, params, create_graph=create_graph, allow_unused=True)
    return loss, list(grads)


def infer_idlg_label(
    named_params: List[Tuple[str, nn.Parameter]],
    target_grads: List[torch.Tensor | None],
    head: str,
) -> int:
    prefix = f"{head}_head"
    bias_candidates = []
    weight_candidates = []
    for (name, param), grad in zip(named_params, target_grads):
        if grad is None or not name.startswith(prefix):
            continue
        if name.endswith("bias") and grad.ndim == 1:
            bias_candidates.append((name, grad.detach()))
        elif name.endswith("weight") and grad.ndim == 2:
            weight_candidates.append((name, grad.detach()))
    if bias_candidates:
        _, bias_grad = bias_candidates[-1]
        return int(torch.argmin(bias_grad).item())
    if not weight_candidates:
        raise ValueError(f"Could not find a final classifier weight gradient for head={head!r}")
    _, weight_grad = weight_candidates[-1]
    return int(torch.argmin(torch.sum(weight_grad, dim=-1)).item())


def gradient_distance(
    dummy_grads: Iterable[torch.Tensor | None],
    target_grads: Iterable[torch.Tensor | None],
) -> torch.Tensor:
    grad_diff = None
    for gx, gy in zip(dummy_grads, target_grads):
        if gx is None or gy is None:
            continue
        if grad_diff is None:
            grad_diff = torch.tensor(0.0, device=gx.device)
        grad_diff = grad_diff + torch.sum((gx - gy) ** 2)
    if grad_diff is None:
        raise ValueError("No overlapping gradients were available for DLG matching")
    return grad_diff


def resolve_dummy_init_scale(value: str | float, reference_x: torch.Tensor) -> Tuple[float, str]:
    raw = str(value).strip().lower()
    if raw == "auto":
        scale = float(reference_x.detach().std().item())
        return max(scale, 1e-12), "auto"
    scale = float(value)
    if scale <= 0:
        raise ValueError("--dummy_init_scale must be > 0 or 'auto'")
    return scale, "fixed"


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x_flat = x.detach().reshape(x.shape[0], -1).cpu()
    y_flat = y.detach().reshape(y.shape[0], -1).cpu()
    vals = []
    for a, b in zip(x_flat, y_flat):
        a = a - a.mean()
        b = b - b.mean()
        denom = torch.norm(a) * torch.norm(b)
        vals.append(float((torch.dot(a, b) / (denom + 1e-12)).item()))
    return float(np.mean(vals))


def add_gradient_laplace_noise(
    named_params: List[Tuple[str, nn.Parameter]],
    grads: List[torch.Tensor | None],
    epsilon: float,
    sensitivity: float,
) -> Tuple[List[torch.Tensor | None], List[Dict], Dict[str, float]]:
    stats = {
        "scale": 0.0,
        "noise_mean": 0.0,
        "noise_std": 0.0,
        "noise_rms": 0.0,
        "grad_rms": 0.0,
        "num_noisy_tensors": 0,
        "num_noisy_elements": 0,
    }
    if epsilon <= 0:
        return grads, [], stats
    if sensitivity <= 0:
        raise ValueError("--trial_laplace_sensitivity must be > 0 when Laplace noise is enabled")
    scale = sensitivity / epsilon
    noisy_grads: List[torch.Tensor | None] = []
    noise_records = []
    flat_noises = []
    flat_grads = []
    for (name, _), grad in zip(named_params, grads):
        if grad is None:
            noisy_grads.append(None)
            continue
        dist = torch.distributions.Laplace(
            loc=torch.zeros((), device=grad.device, dtype=grad.dtype),
            scale=torch.tensor(scale, device=grad.device, dtype=grad.dtype),
        )
        noise = dist.sample(grad.shape)
        noisy_grads.append(grad + noise)
        noise_records.append({"name": name, "noise": noise.detach().cpu()})
        flat_noises.append(noise.detach().reshape(-1).cpu())
        flat_grads.append(grad.detach().reshape(-1).cpu())

    if flat_noises:
        all_noise = torch.cat(flat_noises)
        all_grads = torch.cat(flat_grads)
        stats.update(
            {
                "scale": float(scale),
                "noise_mean": float(all_noise.mean().item()),
                "noise_std": float(all_noise.std(unbiased=False).item()),
                "noise_rms": float(torch.sqrt(torch.mean(all_noise ** 2)).item()),
                "grad_rms": float(torch.sqrt(torch.mean(all_grads ** 2)).item()),
                "num_noisy_tensors": len(flat_noises),
                "num_noisy_elements": int(all_noise.numel()),
            }
        )
    return noisy_grads, noise_records, stats


@torch.no_grad()
def identity_metrics(
    model: nn.Module,
    x: torch.Tensor,
    y_user: torch.Tensor,
    topk: int,
) -> Dict[str, float]:
    model.eval()
    logits = head_logits(model, x, head="user")
    probs = F.softmax(logits, dim=1)
    k = min(topk, probs.shape[1])
    top_idx = probs.topk(k=k, dim=1).indices
    top1 = top_idx[:, 0]
    true_in_topk = (top_idx == y_user[:, None]).any(dim=1)
    true_conf = probs.gather(1, y_user[:, None]).squeeze(1)
    return {
        "user_top1_acc": float((top1 == y_user).float().mean().item()),
        f"user_top{k}_acc": float(true_in_topk.float().mean().item()),
        "true_user_conf": float(true_conf.mean().item()),
    }


@torch.no_grad()
def identity_details(
    model: nn.Module,
    x: torch.Tensor,
    y_user: torch.Tensor,
    topk: int,
) -> List[Dict]:
    model.eval()
    logits = head_logits(model, x, head="user")
    probs = F.softmax(logits, dim=1)
    k = min(topk, probs.shape[1])
    top_probs, top_idx = probs.topk(k=k, dim=1)
    rows = []
    for i in range(x.shape[0]):
        true_user = int(y_user[i].item())
        rows.append(
            {
                "sample_in_batch": i,
                "true_user": true_user,
                "pred_user": int(top_idx[i, 0].item()),
                "true_user_conf": float(probs[i, true_user].item()),
                "topk_users": [int(v) for v in top_idx[i].detach().cpu().tolist()],
                "topk_probs": [float(v) for v in top_probs[i].detach().cpu().tolist()],
                "true_user_in_topk": bool((top_idx[i] == y_user[i]).any().item()),
            }
        )
    return rows


@torch.no_grad()
def task_accuracy(model: nn.Module, x: torch.Tensor, y_task: torch.Tensor) -> float:
    logits = head_logits(model, x, head="task")
    pred = logits.argmax(dim=1)
    return float((pred == y_task).float().mean().item())


@torch.no_grad()
def task_details(model: nn.Module, x: torch.Tensor, y_task: torch.Tensor) -> List[Dict]:
    logits = head_logits(model, x, head="task")
    probs = F.softmax(logits, dim=1)
    pred = logits.argmax(dim=1)
    rows = []
    for i in range(x.shape[0]):
        true_task = int(y_task[i].item())
        rows.append(
            {
                "sample_in_batch": i,
                "true_task": true_task,
                "pred_task": int(pred[i].item()),
                "correct": bool((pred[i] == y_task[i]).item()),
                "true_task_conf": float(probs[i, true_task].item()),
                "task_logits": [float(v) for v in logits[i].detach().cpu().tolist()],
                "task_probs": [float(v) for v in probs[i].detach().cpu().tolist()],
            }
        )
    return rows


def resolve_eval_session_internal(
    ds,
    eval_session_internal: int | None,
    eval_session_original: int | None,
) -> int | None:
    if eval_session_internal is not None and eval_session_original is not None:
        raise ValueError("Use only one of --eval_session_internal or --eval_session_original")
    if eval_session_original is None:
        return eval_session_internal
    if eval_session_original not in ds.session_map:
        raise ValueError(
            f"eval_session_original={eval_session_original} is not in dataset sessions "
            f"{ds.session_original_values}"
        )
    return int(ds.session_map[eval_session_original])


def filter_indices_by_session(ds, indices: List[int], session_internal: int | None, split: str) -> List[int]:
    if session_internal is None:
        return indices
    if session_internal < 0 or session_internal >= ds.n_sessions:
        raise ValueError(
            f"eval_session_internal={session_internal} is out of range for n_sessions={ds.n_sessions}"
        )
    filtered = [idx for idx in indices if int(ds.y_session[idx].item()) == session_internal]
    if not filtered:
        session_original = ds.session_original_values[session_internal]
        raise ValueError(
            f"No samples left after filtering split={split!r} to "
            f"eval_session_internal={session_internal} "
            f"(original={session_original})"
        )
    return filtered


def select_indices(
    ds,
    split: str,
    train_session_internal: int | None,
    batch_size: int,
    seed: int,
    eval_session_internal: int | None = None,
) -> List[int]:
    if split == "all":
        pool = list(range(len(ds)))
    else:
        if train_session_internal is None:
            raise ValueError("--train_session_internal is required when --split is train or test")
        train_idx, test_idx = build_loso_split(ds, train_session_internal)
        pool = train_idx if split == "train" else test_idx

    pool = filter_indices_by_session(ds, pool, eval_session_internal, split)

    if batch_size > len(pool):
        raise ValueError(f"batch_size={batch_size} is larger than selected pool size={len(pool)}")
    rng = random.Random(seed)
    return rng.sample(pool, batch_size)


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_topk_row(row: Dict) -> str:
    pairs = [
        f"user{user}:{prob:.4f}"
        for user, prob in zip(row["topk_users"], row["topk_probs"])
    ]
    hit = "yes" if row["true_user_in_topk"] else "no"
    return (
        f"sample={row['sample_in_batch']} true_user={row['true_user']} "
        f"pred={row['pred_user']} hit_topk={hit} "
        f"true_conf={row['true_user_conf']:.4f} topk=[{', '.join(pairs)}]"
    )


def format_summary_text(summary: Dict) -> str:
    topk = summary["topk"]
    topk_key = f"user_top{topk}_acc"
    lines = [
        "DLG EEG Summary",
        "=" * 80,
        f"dataset          : {summary['dataset']}",
        f"model            : {summary['model']}",
        f"checkpoint       : {summary['checkpoint']}",
        f"checkpoint_epoch : {summary['checkpoint_epoch']}",
        f"attack_head      : {summary['attack_head']} (gradient/iDLG reconstruction)",
        "eval_heads       : task metrics use task_head; UIA metrics use user_head",
        f"label_mode       : {summary['label_info']['label_mode']}",
        f"dummy_init_scale : {summary['dummy_init_scale']} ({summary['dummy_init_mode']})",
        f"grad_loss_scale  : {summary['grad_loss_scale']}",
        f"target_stats     : {summary['target_input_stats']}",
        f"real_task_eval   : {summary['real_task_eval']}",
        f"batch_indices    : {summary['batch_indices']}",
        f"y_task           : {summary['y_task']}",
        f"y_user           : {summary['y_user']}",
        f"trial_laplace    : {summary['trial_laplace']}",
        f"random Top-{topk} : {format_pct(summary['random_topk_baseline'])}",
        "",
        "Aggregate Metrics",
        "-" * 80,
        f"{'input':<8} {'Top-1':>10} {('Top-' + str(topk)):>10} {'TrueConf':>12} {'TaskAcc':>10}",
    ]
    for name in ("real", "recon", "noise"):
        metrics = summary["identity_metrics"][name]
        lines.append(
            f"{name:<8} "
            f"{format_pct(metrics['user_top1_acc']):>10} "
            f"{format_pct(metrics[topk_key]):>10} "
            f"{metrics['true_user_conf']:>12.4f} "
            f"{format_pct(metrics['task_acc']):>10}"
        )

    recon = summary["final_reconstruction"]
    lines.extend(
        [
            "",
            "Reconstruction",
            "-" * 80,
            f"mse              : {recon['mse']:.4f}",
            f"corr             : {recon['corr']:.4f}",
        ]
    )
    if recon.get("mse_to_clean") is not None:
        lines.extend(
            [
                f"mse_to_clean     : {recon['mse_to_clean']:.4f}",
                f"corr_to_clean    : {recon['corr_to_clean']:.4f}",
            ]
        )
    lines.extend(
        [
            "",
            f"Top-{topk} Predictions",
            "-" * 80,
        ]
    )
    for name in ("real", "recon", "noise"):
        lines.append(f"[{name}]")
        for row in summary["identity_details"][name]:
            lines.append(format_topk_row(row))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_topk_csv(summary: Dict) -> str:
    lines = ["input,sample_in_batch,true_user,pred_user,true_user_conf,true_user_in_topk,rank,user,prob"]
    for input_name in ("real", "recon", "noise"):
        for row in summary["identity_details"][input_name]:
            for rank, (user, prob) in enumerate(zip(row["topk_users"], row["topk_probs"]), start=1):
                lines.append(
                    f"{input_name},{row['sample_in_batch']},{row['true_user']},"
                    f"{row['pred_user']},{row['true_user_conf']:.4f},"
                    f"{int(row['true_user_in_topk'])},{rank},{user},{prob:.4f}"
                )
    return "\n".join(lines) + "\n"


def choose_plot_channel(real_x: torch.Tensor, requested_channel: int) -> int:
    if requested_channel >= 0:
        if requested_channel >= real_x.shape[0]:
            raise ValueError(
                f"plot_channel={requested_channel} is out of range for n_channels={real_x.shape[0]}"
            )
        return requested_channel
    return int(torch.var(real_x, dim=1).argmax().item())


def waveform_ylim(target: np.ndarray, recons: List[np.ndarray]) -> Tuple[float, float]:
    values = [target.reshape(-1)] + [recon.reshape(-1) for recon in recons]
    all_values = np.concatenate(values)
    finite_values = all_values[np.isfinite(all_values)]
    if finite_values.size == 0:
        return -1.0, 1.0
    y_min = float(finite_values.min())
    y_max = float(finite_values.max())
    if math.isclose(y_min, y_max):
        pad = max(abs(y_min) * 0.05, 1e-3)
    else:
        pad = (y_max - y_min) * 0.08
    return y_min - pad, y_max + pad


def save_waveform_trajectory(
    real_x: torch.Tensor,
    clean_x: torch.Tensor,
    snapshots: List[Dict],
    path: Path,
    plot_channel: int,
    sfreq: float,
    show_grid: bool,
    font_size: float,
) -> Tuple[int, List[str]]:
    channel = choose_plot_channel(clean_x, plot_channel)
    target = real_x[channel].detach().cpu().numpy()
    recon_series = [snapshot["tensor"][channel].detach().cpu().numpy() for snapshot in snapshots]
    y_limits = waveform_ylim(target, recon_series)
    time_ms = np.arange(real_x.shape[1], dtype=np.float32) / float(sfreq) * 1000.0
    cols = min(3, len(snapshots))
    rows = math.ceil(len(snapshots) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.4 * rows), squeeze=False)
    panel_paths = []
    for ax, snapshot, recon in zip(axes.flat, snapshots, recon_series):
        ax.plot(time_ms, target, linewidth=1.7, color="#111111", label="Target")
        ax.plot(time_ms, recon, linewidth=1.3, color="#d95f02", alpha=0.92, label="Reconstruction")
        ax.set_ylim(*y_limits)
        ax.set_xlabel("Time (ms)", fontsize=font_size)
        ax.set_ylabel("Amplitude (μV)", fontsize=font_size)
        ax.tick_params(axis="both", labelsize=font_size)
        legend = ax.legend(
            fontsize=max(font_size - 1.0, 6.0),
            frameon=True,
            loc="upper right",
            edgecolor="black",
            fancybox=False,
        )
        legend.get_frame().set_linewidth(0.8)
        if show_grid:
            ax.grid(alpha=0.22)

        label = str(snapshot["label"]).replace(" ", "_").replace("/", "_")
        panel_path = path.with_name(f"{path.stem}_{label}{path.suffix}")
        panel_fig, panel_ax = plt.subplots(figsize=(5.2, 3.4))
        panel_ax.plot(time_ms, target, linewidth=1.7, color="#111111", label="Target")
        panel_ax.plot(time_ms, recon, linewidth=1.3, color="#d95f02", alpha=0.92, label="Reconstruction")
        panel_ax.set_ylim(*y_limits)
        panel_ax.set_xlabel("Time (ms)", fontsize=font_size)
        panel_ax.set_ylabel("Amplitude (μV)", fontsize=font_size)
        panel_ax.tick_params(axis="both", labelsize=font_size)
        legend = panel_ax.legend(
            fontsize=max(font_size - 1.0, 6.0),
            frameon=True,
            loc="upper right",
            edgecolor="black",
            fancybox=False,
        )
        legend.get_frame().set_linewidth(0.8)
        if show_grid:
            panel_ax.grid(alpha=0.22)
        panel_fig.tight_layout()
        panel_fig.savefig(panel_path, dpi=180)
        plt.close(panel_fig)
        panel_paths.append(str(panel_path))
    for ax in axes.flat[len(snapshots):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return channel, panel_paths


def run_dlg(args: argparse.Namespace) -> Dict:
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ds = build_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        normalize=args.normalize,
        euclidean_align=args.euclidean_align,
    )

    if args.random_init_model:
        model, checkpoint = build_random_init_model(
            ds=ds,
            model_name=args.model,
            user_hidden_dim=args.user_hidden_dim,
            user_dropout=args.user_dropout,
            device=device,
        )
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required unless --random_init_model is set")
        model, checkpoint = load_checkpoint_model(
            checkpoint_path=Path(args.checkpoint),
            ds=ds,
            model_name=args.model,
            user_hidden_dim=args.user_hidden_dim,
            user_dropout=args.user_dropout,
            device=device,
        )

    ckpt_extra = checkpoint.get("extra", {})
    train_session_internal = args.train_session_internal
    if train_session_internal is None:
        train_session_internal = ckpt_extra.get("train_session_internal")
    eval_session_internal = resolve_eval_session_internal(
        ds,
        args.eval_session_internal,
        args.eval_session_original,
    )

    indices = args.indices
    if indices:
        batch_indices = [int(v.strip()) for v in indices.split(",") if v.strip()]
        if eval_session_internal is not None:
            filtered_indices = filter_indices_by_session(ds, batch_indices, eval_session_internal, args.split)
            if len(filtered_indices) != len(batch_indices):
                raise ValueError(
                    "--indices contains samples outside the requested eval session "
                    f"{eval_session_internal}"
                )
    else:
        batch_indices = select_indices(
            ds,
            split=args.split,
            train_session_internal=train_session_internal,
            batch_size=args.batch_size,
            seed=args.seed,
            eval_session_internal=eval_session_internal,
        )

    if getattr(args, "append_indices_to_out_dir", False):
        trial_tag = "trial" + "p".join(str(idx) for idx in batch_indices)
        out_path = Path(args.out_dir)
        if out_path.name.endswith("_trialrandom"):
            out_path = out_path.with_name(out_path.name.removesuffix("_trialrandom") + f"_{trial_tag}")
        elif out_path.name.endswith("_trial"):
            out_path = out_path.with_name(out_path.name + trial_tag.removeprefix("trial"))
        elif trial_tag not in out_path.name:
            out_path = out_path.with_name(out_path.name + f"_{trial_tag}")
        args.out_dir = str(out_path)

    batch = [ds[i] for i in batch_indices]
    clean_x = torch.stack([item[0] for item in batch], dim=0).to(device)
    y_task = torch.stack([item[1] for item in batch], dim=0).to(device)
    y_user = torch.stack([item[2] for item in batch], dim=0).to(device)
    y_session = torch.stack([item[3] for item in batch], dim=0).to(device)
    attack_labels = y_task if args.attack_head == "task" else y_user
    trial_laplace_epsilon = float(getattr(args, "trial_laplace_epsilon", 0.0) or 0.0)
    trial_laplace_sensitivity = float(getattr(args, "trial_laplace_sensitivity", 1.0))
    x = clean_x

    model.zero_grad(set_to_none=True)
    named_params = named_trainable_parameters(model)
    _, target_grads = gradients_from_batch(
        model=model,
        x=x,
        y=attack_labels,
        head=args.attack_head,
        create_graph=False,
        named_params=named_params,
    )
    target_grads = [None if g is None else g.detach() for g in target_grads]
    target_grads, gradient_laplace_noise, gradient_laplace_stats = add_gradient_laplace_noise(
        named_params,
        target_grads,
        epsilon=trial_laplace_epsilon,
        sensitivity=trial_laplace_sensitivity,
    )

    if args.label_mode == "idlg":
        if x.shape[0] != 1:
            raise ValueError("iDLG label inference is only implemented for batch_size=1")
        inferred = infer_idlg_label(named_params, target_grads, args.attack_head)
        dummy_y = torch.tensor([inferred], dtype=torch.long, device=device)
        dummy_label_logits = None
    elif args.label_mode == "true":
        dummy_y = attack_labels.detach()
        dummy_label_logits = None
    else:
        num_classes = int(head_logits(model, x, head=args.attack_head).shape[1])
        dummy_y = None
        dummy_label_logits = torch.randn(x.shape[0], num_classes, device=device, requires_grad=True)

    dummy_init_scale, dummy_init_mode = resolve_dummy_init_scale(
        getattr(args, "dummy_init_scale", 1.0),
        x,
    )
    grad_loss_scale = float(getattr(args, "grad_loss_scale", 1.0))
    if grad_loss_scale <= 0:
        raise ValueError("--grad_loss_scale must be > 0")
    dummy_x = (torch.randn_like(x) * dummy_init_scale).requires_grad_(True)
    opt_params = [dummy_x] if dummy_label_logits is None else [dummy_x, dummy_label_logits]

    if args.optimizer == "lbfgs":
        line_search_fn = None if args.lbfgs_line_search == "none" else args.lbfgs_line_search
        optimizer = torch.optim.LBFGS(
            opt_params,
            lr=args.lr,
            max_iter=args.lbfgs_max_iter,
            line_search_fn=line_search_fn,
        )
    else:
        optimizer = torch.optim.Adam(opt_params, lr=args.lr)

    no_visualizations = getattr(args, "no_visualizations", False)
    save_artifacts = getattr(args, "save_artifacts", True)
    history = []
    snapshots = []
    if not no_visualizations:
        snapshots.append({"label": "iter 0", "tensor": dummy_x.detach()[0].cpu().clone()})
    out_dir = Path(args.out_dir)
    if save_artifacts or not no_visualizations:
        out_dir.mkdir(parents=True, exist_ok=True)

    for it in range(1, args.iters + 1):
        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            logits = head_logits(model, dummy_x, head=args.attack_head)
            if dummy_label_logits is None:
                loss = F.cross_entropy(logits, dummy_y)
            else:
                loss = soft_cross_entropy(logits, F.softmax(dummy_label_logits, dim=1))
            params = [p for _, p in named_params]
            dummy_grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            grad_loss = gradient_distance(dummy_grads, target_grads)
            (grad_loss * grad_loss_scale).backward()
            return grad_loss

        if args.optimizer == "lbfgs":
            grad_loss = optimizer.step(closure)
        else:
            grad_loss = closure()
            optimizer.step()

        if not torch.isfinite(grad_loss).all() or not torch.isfinite(dummy_x).all():
            raise FloatingPointError(
                "DLG optimization produced NaN/Inf. Try a smaller --lr, "
                "--optimizer adam, or a smaller --dummy_init_scale."
            )

        if it % args.log_every == 0 or it == args.iters:
            with torch.no_grad():
                mse = F.mse_loss(dummy_x, x).item()
                corr = pearson_corr(dummy_x, x)
                history.append(
                    {
                        "iter": it,
                        "grad_loss": float(grad_loss.item()),
                        "mse": float(mse),
                        "corr": float(corr),
                    }
                )
                if not no_visualizations:
                    snapshots.append(
                        {
                            "label": f"iter {it}",
                            "tensor": dummy_x.detach()[0].cpu().clone(),
                        }
                    )
                print(
                    f"[DLG] iter={it:04d} grad_loss={grad_loss.item():.6e} "
                    f"mse={mse:.6e} corr={corr:.4f}"
                )

    recon = dummy_x.detach()
    noise = torch.randn_like(x)
    topk = min(args.topk, ds.n_users)
    metrics = {
        "recon": identity_metrics(model, recon, y_user, topk),
        "real": identity_metrics(model, x, y_user, topk),
        "noise": identity_metrics(model, noise, y_user, topk),
    }
    metrics["recon"]["task_acc"] = task_accuracy(model, recon, y_task)
    metrics["real"]["task_acc"] = task_accuracy(model, x, y_task)
    metrics["noise"]["task_acc"] = task_accuracy(model, noise, y_task)
    with torch.no_grad():
        real_task_logits = head_logits(model, x, head="task")
        real_task_probs = F.softmax(real_task_logits, dim=1)
        real_task_loss = F.cross_entropy(real_task_logits, y_task)
        real_task_pred = real_task_probs.argmax(dim=1)

    label_info = {
        "attack_labels": attack_labels.detach().cpu().tolist(),
        "label_mode": args.label_mode,
    }
    if dummy_label_logits is not None:
        label_info["optimized_labels"] = dummy_label_logits.detach().argmax(dim=1).cpu().tolist()
    elif args.label_mode == "idlg":
        label_info["idlg_inferred_labels"] = dummy_y.detach().cpu().tolist()

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "checkpoint": None if args.random_init_model else str(args.checkpoint),
        "random_init_model": bool(args.random_init_model),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_extra": ckpt_extra,
        "attack_head": args.attack_head,
        "dummy_init_scale": dummy_init_scale,
        "dummy_init_mode": dummy_init_mode,
        "grad_loss_scale": grad_loss_scale,
        "target_input_stats": {
            "min": float(x.min().item()),
            "max": float(x.max().item()),
            "mean": float(x.mean().item()),
            "std": float(x.std().item()),
        },
        "real_task_eval": {
            "loss": float(real_task_loss.item()),
            "pred": real_task_pred.detach().cpu().tolist(),
            "true_conf": real_task_probs.gather(1, y_task.view(-1, 1)).squeeze(1).detach().cpu().tolist(),
            "correct": (real_task_pred == y_task).detach().cpu().tolist(),
            "logits": real_task_logits.detach().cpu().tolist(),
        },
        "batch_indices": batch_indices,
        "train_session_internal": train_session_internal,
        "eval_session_internal": eval_session_internal,
        "eval_session_original": None
        if eval_session_internal is None
        else ds.session_original_values[eval_session_internal],
        "session_original_values": ds.session_original_values,
        "split": args.split,
        "topk": topk,
        "random_topk_baseline": float(topk / ds.n_users),
        "y_task": y_task.detach().cpu().tolist(),
        "y_user": y_user.detach().cpu().tolist(),
        "y_session": y_session.detach().cpu().tolist(),
        "trial_laplace": {
            "enabled": bool(trial_laplace_epsilon > 0),
            "applied_to": "gradients",
            "epsilon": trial_laplace_epsilon,
            "sensitivity": trial_laplace_sensitivity,
            "scale": gradient_laplace_stats["scale"],
            "noise_mean": gradient_laplace_stats["noise_mean"],
            "noise_std": gradient_laplace_stats["noise_std"],
            "noise_rms": gradient_laplace_stats["noise_rms"],
            "grad_rms": gradient_laplace_stats["grad_rms"],
            "num_noisy_tensors": gradient_laplace_stats["num_noisy_tensors"],
            "num_noisy_elements": gradient_laplace_stats["num_noisy_elements"],
            "target_mse_to_clean": float(F.mse_loss(x, clean_x).item()),
            "target_corr_to_clean": pearson_corr(x, clean_x),
        },
        "label_info": label_info,
        "final_reconstruction": {
            "mse": float(F.mse_loss(recon, x).item()),
            "corr": pearson_corr(recon, x),
            "mse_to_clean": None
            if trial_laplace_epsilon <= 0
            else float(F.mse_loss(recon, clean_x).item()),
            "corr_to_clean": None
            if trial_laplace_epsilon <= 0
            else pearson_corr(recon, clean_x),
        },
        "identity_metrics": metrics,
        "history": history,
    }

    detail_rows = {
        "recon": identity_details(model, recon, y_user, topk),
        "real": identity_details(model, x, y_user, topk),
        "noise": identity_details(model, noise, y_user, topk),
    }
    summary["identity_details"] = detail_rows
    summary["task_details"] = {
        "recon": task_details(model, recon, y_task),
        "real": task_details(model, x, y_task),
        "noise": task_details(model, noise, y_task),
    }
    if not no_visualizations:
        waveform_path = out_dir / "eeg_waveform_trajectory.png"
        plot_channel, panel_paths = save_waveform_trajectory(
            real_x=x.detach()[0].cpu(),
            clean_x=clean_x.detach()[0].cpu(),
            snapshots=snapshots,
            path=waveform_path,
            plot_channel=args.plot_channel,
            sfreq=args.sfreq,
            show_grid=args.waveform_grid,
            font_size=args.waveform_font_size,
        )
        summary["visualization"] = {
            "plot_channel": plot_channel,
            "waveform_grid": args.waveform_grid,
            "waveform_font_size": args.waveform_font_size,
            "waveform_trajectory_png": str(waveform_path),
            "waveform_panel_pngs": panel_paths,
        }
    else:
        summary["visualization"] = None

    if save_artifacts:
        save_json(summary, out_dir / "summary.json")
        save_text(format_summary_text(summary), out_dir / "summary.txt")
        save_text(format_topk_csv(summary), out_dir / "topk_predictions.csv")
        torch.save(
            {
                "real_x": x.detach().cpu(),
                "clean_x": clean_x.detach().cpu(),
                "recon_x": recon.detach().cpu(),
                "noise_x": noise.detach().cpu(),
                "gradient_laplace_noise": gradient_laplace_noise,
                "y_task": y_task.detach().cpu(),
                "y_user": y_user.detach().cpu(),
                "y_session": y_session.detach().cpu(),
                "batch_indices": batch_indices,
                "history": history,
                "snapshots": snapshots,
            },
            out_dir / "reconstruction.pt",
        )
        print(f"Saved {out_dir / 'summary.json'}")
        print(f"Saved {out_dir / 'summary.txt'}")
        print(f"Saved {out_dir / 'topk_predictions.csv'}")
        print(f"Saved {out_dir / 'reconstruction.pt'}")
        if summary["visualization"] is not None:
            print(f"Saved {summary['visualization']['waveform_trajectory_png']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple DLG/iDLG attack for EEG checkpoints or a shared initialized model")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--random_init_model",
        action="store_true",
        help="Use a seeded randomly initialized model instead of loading --checkpoint.",
    )
    parser.add_argument("--dataset", type=str, default="P300")
    parser.add_argument("--data_dir", type=str, default="data/P300")
    parser.add_argument("--model", type=str, default="EEGNet", choices=["EEGNet", "LMEEGNet", "ShallowCNN", "Linear"])
    parser.add_argument("--normalize", type=str, default="channel", choices=["none", "trial", "channel"])
    parser.add_argument("--euclidean_align", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"])
    parser.add_argument("--train_session_internal", type=int, default=None)
    parser.add_argument(
        "--eval_session_internal",
        type=int,
        default=None,
        help="Optionally restrict DLG sample selection to one internal session id.",
    )
    parser.add_argument(
        "--eval_session_original",
        type=int,
        default=None,
        help="Optionally restrict DLG sample selection to one original session label.",
    )
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--attack_head",
        type=str,
        default="task",
        choices=["task", "user"],
        help=(
            "Head used to generate and match attack gradients. Reconstruction is evaluated "
            "with both heads: task_head for task metrics and user_head for UIA metrics."
        ),
    )
    parser.add_argument("--label_mode", type=str, default="idlg", choices=["idlg", "true", "dlg"])
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="lbfgs", choices=["lbfgs", "adam"])
    parser.add_argument("--lbfgs_max_iter", type=int, default=20)
    parser.add_argument("--lbfgs_line_search", type=str, default="strong_wolfe", choices=["none", "strong_wolfe"])
    parser.add_argument("--grad_loss_scale", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--dummy_init_scale",
        type=str,
        default="1.0",
        help="Standard deviation used to initialize the dummy EEG tensor, or 'auto' to use the target sample std.",
    )
    parser.add_argument(
        "--trial_laplace_epsilon",
        type=float,
        default=0.0,
        help="If > 0, add Laplace noise to the locally computed gradients before DLG matching.",
    )
    parser.add_argument(
        "--trial_laplace_sensitivity",
        type=float,
        default=1.0,
        help="Sensitivity used for trial Laplace noise scale = sensitivity / epsilon.",
    )
    parser.add_argument("--user_hidden_dim", type=int, default=256)
    parser.add_argument("--user_dropout", type=float, default=0.5)
    parser.add_argument(
        "--plot_channel",
        type=int,
        default=-1,
        help="Channel index for waveform panels; -1 picks the target channel with the largest variance.",
    )
    parser.add_argument("--sfreq", type=float, default=128.0, help="Sampling rate used for waveform x-axis in ms.")
    parser.add_argument(
        "--waveform_grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show background grid lines on waveform plots.",
    )
    parser.add_argument("--no_waveform_grid", dest="waveform_grid", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument(
        "--waveform_font_size",
        type=float,
        default=10.0,
        help="Font size for waveform axis labels and tick labels.",
    )
    parser.add_argument("--out_dir", type=str, default="checkpoint/dlg_attack")
    parser.add_argument(
        "--append_indices_to_out_dir",
        action="store_true",
        help="Append selected trial indices to out_dir after random selection is resolved.",
    )
    parser.add_argument("--no_visualizations", action="store_true")
    parser.add_argument("--no_artifacts", action="store_true")
    args = parser.parse_args()
    args.save_artifacts = not args.no_artifacts
    run_dlg(args)


if __name__ == "__main__":
    main()

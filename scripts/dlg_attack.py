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
    candidates = []
    prefix = f"{head}_head"
    for (name, param), grad in zip(named_params, target_grads):
        if grad is not None and name.startswith(prefix) and name.endswith("weight") and grad.ndim == 2:
            candidates.append((name, grad.detach()))
    if not candidates:
        raise ValueError(f"Could not find a final classifier weight gradient for head={head!r}")
    _, weight_grad = candidates[-1]
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


def select_indices(ds, split: str, train_session_internal: int | None, batch_size: int, seed: int) -> List[int]:
    if split == "all":
        pool = list(range(len(ds)))
    else:
        if train_session_internal is None:
            raise ValueError("--train_session_internal is required when --split is train or test")
        train_idx, test_idx = build_loso_split(ds, train_session_internal)
        pool = train_idx if split == "train" else test_idx

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
        f"attack_head      : {summary['attack_head']}",
        f"label_mode       : {summary['label_info']['label_mode']}",
        f"batch_indices    : {summary['batch_indices']}",
        f"y_task           : {summary['y_task']}",
        f"y_user           : {summary['y_user']}",
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


def save_waveform_trajectory(
    real_x: torch.Tensor,
    snapshots: List[Dict],
    path: Path,
    plot_channel: int,
    sfreq: float,
) -> int:
    channel = choose_plot_channel(real_x, plot_channel)
    target = real_x[channel].detach().cpu().numpy()
    time_ms = np.arange(real_x.shape[1], dtype=np.float32) / float(sfreq) * 1000.0
    cols = min(3, len(snapshots))
    rows = math.ceil(len(snapshots) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.4 * rows), squeeze=False)
    for ax, snapshot in zip(axes.flat, snapshots):
        recon = snapshot["tensor"][channel].detach().cpu().numpy()
        ax.plot(time_ms, target, label="target", linewidth=1.7, color="#111111")
        ax.plot(time_ms, recon, label="recon", linewidth=1.3, color="#d95f02", alpha=0.92)
        ax.set_title(f"{snapshot['label']} | channel {channel}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (a.u.)")
        ax.grid(alpha=0.22)
    for ax in axes.flat[len(snapshots):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("single-channel EEG waveform reconstruction trajectory", y=0.985)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return channel


def run_dlg(args: argparse.Namespace) -> Dict:
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = build_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        normalize=args.normalize,
        euclidean_align=args.euclidean_align,
    )

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

    indices = args.indices
    if indices:
        batch_indices = [int(v.strip()) for v in indices.split(",") if v.strip()]
    else:
        batch_indices = select_indices(
            ds,
            split=args.split,
            train_session_internal=train_session_internal,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    batch = [ds[i] for i in batch_indices]
    x = torch.stack([item[0] for item in batch], dim=0).to(device)
    y_task = torch.stack([item[1] for item in batch], dim=0).to(device)
    y_user = torch.stack([item[2] for item in batch], dim=0).to(device)
    attack_labels = y_task if args.attack_head == "task" else y_user

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

    dummy_x = torch.randn_like(x, requires_grad=True)
    opt_params = [dummy_x] if dummy_label_logits is None else [dummy_x, dummy_label_logits]

    if args.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(
            opt_params,
            lr=args.lr,
            max_iter=20,
            line_search_fn="strong_wolfe",
        )
    else:
        optimizer = torch.optim.Adam(opt_params, lr=args.lr)

    history = []
    snapshots = [{"label": "iter 0", "tensor": dummy_x.detach()[0].cpu().clone()}]
    out_dir = Path(args.out_dir)
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
            grad_loss.backward()
            return grad_loss

        if args.optimizer == "lbfgs":
            grad_loss = optimizer.step(closure)
        else:
            grad_loss = closure()
            optimizer.step()

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
                snapshots.append(
                    {
                        "label": f"iter {it}",
                        "tensor": dummy_x.detach()[0].cpu().clone(),
                    }
                )
                print(
                    f"[DLG] iter={it:04d} grad_loss={grad_loss.item():.6f} "
                    f"mse={mse:.6f} corr={corr:.4f}"
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
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_extra": ckpt_extra,
        "attack_head": args.attack_head,
        "batch_indices": batch_indices,
        "train_session_internal": train_session_internal,
        "split": args.split,
        "topk": topk,
        "random_topk_baseline": float(topk / ds.n_users),
        "y_task": y_task.detach().cpu().tolist(),
        "y_user": y_user.detach().cpu().tolist(),
        "label_info": label_info,
        "final_reconstruction": {
            "mse": float(F.mse_loss(recon, x).item()),
            "corr": pearson_corr(recon, x),
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
    waveform_path = out_dir / "eeg_waveform_trajectory.png"
    plot_channel = save_waveform_trajectory(
        real_x=x.detach()[0].cpu(),
        snapshots=snapshots,
        path=waveform_path,
        plot_channel=args.plot_channel,
        sfreq=args.sfreq,
    )
    summary["visualization"] = {
        "plot_channel": plot_channel,
        "waveform_trajectory_png": str(waveform_path),
    }
    save_json(summary, out_dir / "summary.json")
    save_text(format_summary_text(summary), out_dir / "summary.txt")
    save_text(format_topk_csv(summary), out_dir / "topk_predictions.csv")
    torch.save(
        {
            "real_x": x.detach().cpu(),
            "recon_x": recon.detach().cpu(),
            "noise_x": noise.detach().cpu(),
            "y_task": y_task.detach().cpu(),
            "y_user": y_user.detach().cpu(),
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
    print(f"Saved {waveform_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple DLG/iDLG attack for EEG checkpoints")
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
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--attack_head", type=str, default="task", choices=["task", "user"])
    parser.add_argument("--label_mode", type=str, default="idlg", choices=["idlg", "true", "dlg"])
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="lbfgs", choices=["lbfgs", "adam"])
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--user_hidden_dim", type=int, default=256)
    parser.add_argument("--user_dropout", type=float, default=0.5)
    parser.add_argument(
        "--plot_channel",
        type=int,
        default=-1,
        help="Channel index for waveform panels; -1 picks the target channel with the largest variance.",
    )
    parser.add_argument("--sfreq", type=float, default=128.0, help="Sampling rate used for waveform x-axis in ms.")
    parser.add_argument("--out_dir", type=str, default="checkpoint/dlg_attack")
    args = parser.parse_args()
    run_dlg(args)


if __name__ == "__main__":
    main()

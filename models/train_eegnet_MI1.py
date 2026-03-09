from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.io as scio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from eegnet import EEGNetConfig, EEGNetMI1MI2
from eval_metrics import evaluate_metrics


# =========================
# 1) Dataset
# =========================
class MI1Dataset(Dataset):
    """
    Load all .mat files in MI1 directory.
    Each file is treated as one user/domain unit.
    """

    def __init__(self, mi1_dir: str):
        mi1_dir = Path(mi1_dir)
        mat_files = sorted(mi1_dir.glob("*.mat"))

        if len(mat_files) == 0:
            raise ValueError(f"No .mat files found in {mi1_dir}")

        X_all = []
        y_all = []
        user_all = []

        print("Loaded mat files:")
        for f in mat_files:
            print(" ", f.name)

        for mat_idx, mat_path in enumerate(mat_files):
            data = scio.loadmat(mat_path)

            X = data["X"]
            y_task = np.asarray(data["y"]).squeeze()
            y_user = np.asarray(data["group"]).squeeze()  # 这里只做一致性检查

            if not (X.shape[0] == len(y_task) == len(y_user)):
                raise ValueError(f"Mismatch in {mat_path}")

            X_all.append(X)
            y_all.append(y_task)

            # 这里把“文件编号”作为用户标签，避免不同文件里的 group 编号重复
            user_all.append(np.full(len(y_task), mat_idx, dtype=np.int64))

        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
        user_all = np.concatenate(user_all, axis=0)

        # 若任务标签不是从0开始，重映射到 0...K-1
        uniq_task = sorted(np.unique(y_all).tolist())
        task_map = {v: i for i, v in enumerate(uniq_task)}
        y_all = np.array([task_map[v] for v in y_all], dtype=np.int64)

        self.x = torch.tensor(X_all, dtype=torch.float32)
        self.y_task = torch.tensor(y_all, dtype=torch.long)
        self.y_user = torch.tensor(user_all, dtype=torch.long)

        self.n_samples = self.x.shape[0]
        self.n_channels = self.x.shape[1]
        self.n_times = self.x.shape[2]
        self.n_task_classes = int(torch.unique(self.y_task).numel())
        self.n_users = int(torch.unique(self.y_user).numel())

        print("\nDataset loaded:")
        print("samples:", self.n_samples)
        print("channels:", self.n_channels)
        print("times:", self.n_times)
        print("task classes:", self.n_task_classes)
        print("users:", self.n_users)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y_task[idx], self.y_user[idx]


# =========================
# 2) Utils
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_dir(save_root: str, run_name: str | None) -> Path:
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    if run_name is None or run_name.strip() == "":
        existing = [p.name for p in save_root.iterdir() if p.is_dir()]
        idx = 1
        while f"run_{idx:03d}" in existing:
            idx += 1
        run_name = f"run_{idx:03d}"

    run_dir = save_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_model(ds: MI1Dataset, user_hidden_dim: int) -> Tuple[EEGNetMI1MI2, EEGNetConfig]:
    cfg = EEGNetConfig(
        n_channels=ds.n_channels,
        n_times=ds.n_times,
    )
    model = EEGNetMI1MI2(
        cfg,
        n_task_classes=ds.n_task_classes,
        n_users=ds.n_users,
        user_hidden_dim=user_hidden_dim,
    )
    return model, cfg


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    cfg: EEGNetConfig,
    extra: Dict,
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "eegnet_config": asdict(cfg),
        "extra": extra,
    }
    torch.save(ckpt, path)


# =========================
# 3) Train / Val
# =========================
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float,
) -> Dict[str, float]:
    model.train()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_task_loss = 0.0
    total_user_loss = 0.0
    total_samples = 0

    for x, y_task, y_user in dataloader:
        x = x.to(device)
        y_task = y_task.to(device)
        y_user = y_user.to(device)

        optimizer.zero_grad()

        task_logits, user_logits = model(x)

        loss_task = ce(task_logits, y_task)
        loss_user = ce(user_logits, y_user)
        loss = loss_task + alpha * loss_user

        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_samples += bs
        total_loss += loss.item() * bs
        total_task_loss += loss_task.item() * bs
        total_user_loss += loss_user.item() * bs

    return {
        "loss": total_loss / total_samples,
        "task_loss": total_task_loss / total_samples,
        "user_loss": total_user_loss / total_samples,
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    alpha: float,
    n_task_classes: int,
) -> Dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_task_loss = 0.0
    total_user_loss = 0.0
    total_samples = 0

    for x, y_task, y_user in dataloader:
        x = x.to(device)
        y_task = y_task.to(device)
        y_user = y_user.to(device)

        task_logits, user_logits = model(x)

        loss_task = ce(task_logits, y_task)
        loss_user = ce(user_logits, y_user)
        loss = loss_task + alpha * loss_user

        bs = x.size(0)
        total_samples += bs
        total_loss += loss.item() * bs
        total_task_loss += loss_task.item() * bs
        total_user_loss += loss_user.item() * bs

    stats = {
        "loss": total_loss / total_samples,
        "task_loss": total_task_loss / total_samples,
        "user_loss": total_user_loss / total_samples,
    }

    metric_stats = evaluate_metrics(
        model,
        dataloader,
        n_task_classes=n_task_classes,
        device=device,
        amp=False,
    )

    stats["mi_acc"] = metric_stats.mi_acc
    stats["user_acc"] = metric_stats.user_acc
    stats["bca"] = metric_stats.bca
    return stats


# =========================
# 4) Main training
# =========================
def main():
    parser = argparse.ArgumentParser(description="Train EEGNet on MI1 with task+user loss")
    parser.add_argument("--mi1_dir", type=str, default="../data/MI1_partial", help="Directory containing MI1 .mat files")
    parser.add_argument("--save_root", type=str, default="./checkpoints_MI1", help="Root dir for runs")
    parser.add_argument("--run_name", type=str, default=None, help="Name of this run")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for user CE loss")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--user_hidden_dim", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=10, help="Save epoch checkpoint every N epochs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- dataset ----
    ds = MI1Dataset(args.mi1_dir)

    print("\n=== Dataset Summary ===")
    print(f"mi1_dir          : {args.mi1_dir}")
    print(f"n_samples        : {ds.n_samples}")
    print(f"n_channels       : {ds.n_channels}")
    print(f"n_times          : {ds.n_times}")
    print(f"n_task_classes   : {ds.n_task_classes}")
    print(f"n_users          : {ds.n_users}")
    print(f"task labels      : {torch.unique(ds.y_task).tolist()}")
    print(f"user labels      : {torch.unique(ds.y_user).tolist()}")
    print()

    # ---- split ----
    n_total = len(ds)
    n_val = max(1, int(round(n_total * args.val_ratio)))
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError("Train split is empty; reduce val_ratio.")

    train_set, val_set = random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ---- model ----
    model, cfg = build_model(ds, user_hidden_dim=args.user_hidden_dim)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ---- save dir ----
    run_dir = make_run_dir(args.save_root, args.run_name)
    print(f"Run directory     : {run_dir}")
    print(f"Device            : {device}")
    print()

    config_to_save = {
        "mi1_dir": args.mi1_dir,
        "save_root": args.save_root,
        "run_name": run_dir.name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "alpha": args.alpha,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "user_hidden_dim": args.user_hidden_dim,
        "save_every": args.save_every,
        "dataset_summary": {
            "n_samples": ds.n_samples,
            "n_channels": ds.n_channels,
            "n_times": ds.n_times,
            "n_task_classes": ds.n_task_classes,
            "n_users": ds.n_users,
        },
        "eegnet_config": asdict(cfg),
    }
    save_json(config_to_save, run_dir / "config.json")

    best_val_loss = float("inf")
    best_val_mi_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha=args.alpha,
        )

        val_stats = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            alpha=args.alpha,
            n_task_classes=ds.n_task_classes,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_task_loss": train_stats["task_loss"],
            "train_user_loss": train_stats["user_loss"],
            "val_loss": val_stats["loss"],
            "val_task_loss": val_stats["task_loss"],
            "val_user_loss": val_stats["user_loss"],
            "val_mi_acc": val_stats["mi_acc"],
            "val_user_acc": val_stats["user_acc"],
            "val_bca": val_stats["bca"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.4f} "
            f"(task={row['train_task_loss']:.4f}, user={row['train_user_loss']:.4f}) | "
            f"val_loss={row['val_loss']:.4f} "
            f"(task={row['val_task_loss']:.4f}, user={row['val_user_loss']:.4f}) | "
            f"val_MI_ACC={row['val_mi_acc'] * 100:.2f}% | "
            f"val_USER_ACC={row['val_user_acc'] * 100:.2f}% | "
            f"val_BCA={row['val_bca'] * 100:.2f}%"
        )

        extra = {
            "alpha": args.alpha,
            "run_name": run_dir.name,
            "dataset_type": "MI1",
        }

        save_checkpoint(
            path=run_dir / "latest.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=row,
            cfg=cfg,
            extra=extra,
        )

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                path=run_dir / f"epoch_{epoch:03d}.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=row,
                cfg=cfg,
                extra=extra,
            )

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            save_checkpoint(
                path=run_dir / "best_by_val_loss.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=row,
                cfg=cfg,
                extra=extra,
            )
            print("  -> saved best_by_val_loss.pth")

        if row["val_mi_acc"] > best_val_mi_acc:
            best_val_mi_acc = row["val_mi_acc"]
            save_checkpoint(
                path=run_dir / "best_by_mi_acc.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=row,
                cfg=cfg,
                extra=extra,
            )
            print("  -> saved best_by_mi_acc.pth")

        save_json({"history": history}, run_dir / "history.json")

    print("\nTraining finished.")
    print(f"Run dir: {run_dir}")
    print("Saved checkpoints:")
    print(f"  - {run_dir / 'latest.pth'}")
    print(f"  - {run_dir / 'best_by_val_loss.pth'}")
    print(f"  - {run_dir / 'best_by_mi_acc.pth'}")
    if args.save_every > 0:
        print(f"  - periodic epoch_XXX.pth files every {args.save_every} epochs")


if __name__ == "__main__":
    main()
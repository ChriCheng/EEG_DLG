from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import scipy.io as scio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from dataclasses import asdict

from models.EEGNet import EEGNetConfig, EEGNetMI1MI2, UserClassifier
from models.ShallowCNN import ShallowCNNConfig, ShallowCNNMI1MI2


# =========================
# 1) Dataset
# =========================
class MI1Dataset(Dataset):
    """
    Load all .mat files in MI1 directory.
    Each .mat file is treated as one user.
    Session is read from each mat file and used for LOSO splitting.
    """

    def __init__(self, mi1_dir: str, normalize: str = "channel"):
        mi1_dir = Path(mi1_dir)
        mat_files = sorted(mi1_dir.glob("*.mat"))

        if len(mat_files) == 0:
            raise ValueError(f"No .mat files found in {mi1_dir}")

        X_all = []
        y_task_all = []
        y_user_all = []
        y_session_all = []

        print("Loaded mat files:")
        for f in mat_files:
            print(" ", f.name)

        for mat_idx, mat_path in enumerate(mat_files):
            data = scio.loadmat(mat_path)

            if "X" not in data or "y" not in data or "session" not in data:
                raise ValueError(f"Missing X/y/session in {mat_path}")

            X = np.asarray(data["X"], dtype=np.float32)   # (N, C, T)
            y_task = np.asarray(data["y"]).reshape(-1)
            y_session = np.asarray(data["session"]).reshape(-1)

            # session 可能是 object / str，先统一转成 int
            y_session = np.array([int(v) for v in y_session], dtype=np.int64)

            if not (X.shape[0] == len(y_task) == len(y_session)):
                raise ValueError(
                    f"Mismatch in {mat_path}: "
                    f"X.shape[0]={X.shape[0]}, len(y_task)={len(y_task)}, len(y_session)={len(y_session)}"
                )

            # ---------- optional normalization ----------
            if normalize == "trial":
                # each trial over all channels + times
                mean = X.mean(axis=(1, 2), keepdims=True)
                std = X.std(axis=(1, 2), keepdims=True) + 1e-6
                X = (X - mean) / std
            elif normalize == "channel":
                # each trial, each channel over time
                mean = X.mean(axis=2, keepdims=True)
                std = X.std(axis=2, keepdims=True) + 1e-6
                X = (X - mean) / std
            elif normalize == "none":
                pass
            else:
                raise ValueError(f"Unknown normalize={normalize}")

            X_all.append(X)
            y_task_all.append(y_task)

            # 每个 mat 文件视作一个用户
            y_user_all.append(np.full(len(y_task), mat_idx, dtype=np.int64))
            y_session_all.append(y_session)

        X_all = np.concatenate(X_all, axis=0)
        y_task_all = np.concatenate(y_task_all, axis=0)
        y_user_all = np.concatenate(y_user_all, axis=0)
        y_session_all = np.concatenate(y_session_all, axis=0)

        # task label remap to 0...K-1
        uniq_task = sorted(np.unique(y_task_all).tolist())
        task_map = {v: i for i, v in enumerate(uniq_task)}
        y_task_all = np.array([task_map[v] for v in y_task_all], dtype=np.int64)

        # session remap to internal 0...S-1
        uniq_session = sorted(np.unique(y_session_all).tolist())
        session_map = {v: i for i, v in enumerate(uniq_session)}
        y_session_internal = np.array([session_map[v] for v in y_session_all], dtype=np.int64)

        self.x = torch.tensor(X_all, dtype=torch.float32)
        self.y_task = torch.tensor(y_task_all, dtype=torch.long)
        self.y_user = torch.tensor(y_user_all, dtype=torch.long)
        self.y_session = torch.tensor(y_session_internal, dtype=torch.long)

        self.session_original_values = uniq_session
        self.session_map = session_map

        self.n_samples = self.x.shape[0]
        self.n_channels = self.x.shape[1]
        self.n_times = self.x.shape[2]
        self.n_task_classes = int(torch.unique(self.y_task).numel())
        self.n_users = int(torch.unique(self.y_user).numel())
        self.n_sessions = int(torch.unique(self.y_session).numel())

        print("\nDataset loaded:")
        print("samples:", self.n_samples)
        print("channels:", self.n_channels)
        print("times:", self.n_times)
        print("task classes:", self.n_task_classes)
        print("users:", self.n_users)
        print("sessions:", self.n_sessions)
        print("original session values:", self.session_original_values)
        print("normalize:", normalize)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y_task[idx],
            self.y_user[idx],
            self.y_session[idx],
        )


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


def build_dataset(dataset_name: str, data_dir: str, normalize: str) -> Dataset:
    dataset_name = dataset_name.strip()

    if dataset_name == "MI1":
        return MI1Dataset(data_dir, normalize=normalize)

    raise NotImplementedError(
        f"Dataset {dataset_name!r} is not implemented yet. "
        f"Currently supported: ['MI1']"
    )


def build_model(ds: Dataset, user_hidden_dim: int, model_name: str) -> Tuple[nn.Module, Any]:
    model_name = model_name.strip()

    if model_name == "EEGNet":
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

    if model_name == "ShallowCNN":
        cfg = ShallowCNNConfig(
            n_channels=ds.n_channels,
            n_times=ds.n_times,
        )
        model = ShallowCNNMI1MI2(
            cfg,
            n_task_classes=ds.n_task_classes,
            n_users=ds.n_users,
            user_hidden_dim=user_hidden_dim,
        )
        return model, cfg

    raise ValueError(
        f"Unknown model={model_name!r}. "
        f"Currently supported: ['EEGNet', 'ShallowCNN']"
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: Dict,
    cfg: Any,
    extra: Dict,
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "metrics": metrics,
        "model_config": asdict(cfg),
        "extra": extra,
    }
    torch.save(ckpt, path)


def build_loso_split(ds: Dataset, train_session_internal: int) -> Tuple[List[int], List[int]]:
    all_sessions = ds.y_session.numpy()
    train_idx = np.where(all_sessions == train_session_internal)[0].tolist()
    test_idx = np.where(all_sessions != train_session_internal)[0].tolist()

    if len(train_idx) == 0:
        raise ValueError(f"No samples for train_session_internal={train_session_internal}")
    if len(test_idx) == 0:
        raise ValueError(f"No samples left for test when training on session {train_session_internal}")

    return train_idx, test_idx


def compute_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(f"y_true/y_pred must be 1-D, got {y_true.shape}, {y_pred.shape}")
    if y_true.numel() != y_pred.numel():
        raise ValueError("y_true and y_pred must have same length")
    if y_true.numel() == 0:
        return 0.0
    return (y_true.to(torch.long) == y_pred.to(torch.long)).float().mean().item()


def compute_bca(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    n_classes: int,
    *,
    ignore_empty_classes: bool = True,
) -> float:
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(f"y_true/y_pred must be 1-D, got {y_true.shape}, {y_pred.shape}")
    if y_true.numel() != y_pred.numel():
        raise ValueError("y_true and y_pred must have same length")

    y_true = y_true.to(torch.long)
    y_pred = y_pred.to(torch.long)

    correct = (y_true == y_pred).to(torch.long)
    total_per_class = torch.bincount(y_true, minlength=n_classes).to(torch.float32)
    correct_per_class = torch.bincount(
        y_true,
        weights=correct.to(torch.float32),
        minlength=n_classes,
    )

    if ignore_empty_classes:
        mask = total_per_class > 0
        if mask.sum().item() == 0:
            return 0.0
        per_class_recall = correct_per_class[mask] / total_per_class[mask]
        return per_class_recall.mean().item()
    else:
        per_class_recall = torch.zeros_like(total_per_class)
        nonzero = total_per_class > 0
        per_class_recall[nonzero] = correct_per_class[nonzero] / total_per_class[nonzero]
        return per_class_recall.mean().item()


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


# =========================
# 3) Stage 1: train task model
# =========================
def train_task_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    all_true = []
    all_pred = []

    for x, y_task, _, _ in dataloader:
        x = x.to(device)
        y_task = y_task.to(device)

        optimizer.zero_grad()

        task_logits = model(x, head="task")
        loss = ce(task_logits, y_task)

        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_samples += bs
        total_loss += loss.item() * bs

        pred = task_logits.argmax(dim=1)
        all_true.append(y_task.detach().cpu())
        all_pred.append(pred.detach().cpu())

    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)

    return {
        "loss": total_loss / total_samples,
        "mi_acc": compute_accuracy(y_true, y_pred),
    }


@torch.no_grad()
def evaluate_task(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_task_classes: int,
) -> Dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    all_true = []
    all_pred = []

    for x, y_task, _, _ in dataloader:
        x = x.to(device)
        y_task = y_task.to(device)

        task_logits = model(x, head="task")
        loss = ce(task_logits, y_task)

        bs = x.size(0)
        total_samples += bs
        total_loss += loss.item() * bs

        pred = task_logits.argmax(dim=1)
        all_true.append(y_task.detach().cpu())
        all_pred.append(pred.detach().cpu())

    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)

    mi_acc = compute_accuracy(y_true, y_pred)
    bca = compute_bca(y_true, y_pred, n_classes=n_task_classes)

    return {
        "loss": total_loss / total_samples,
        "mi_acc": mi_acc,
        "bca": bca,
    }


# =========================
# 4) Stage 2: train user classifier on fixed backbone
# =========================
def reset_user_head(
    model: nn.Module,
    n_users: int,
    hidden_dim: int,
    device: torch.device,
    dropout: float 
) -> None:
    model.user_head = UserClassifier(
        feature_dim=model.backbone.feature_dim,
        n_users=n_users,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)


def train_user_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    # backbone fixed, user_head train
    model.backbone.eval()
    model.task_head.eval()
    model.user_head.train()

    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    all_true = []
    all_pred = []

    for x, _, y_user, _ in dataloader:
        x = x.to(device)
        y_user = y_user.to(device)

        optimizer.zero_grad()

        user_logits = model(x, head="user")
        loss = ce(user_logits, y_user)

        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_samples += bs
        total_loss += loss.item() * bs

        pred = user_logits.argmax(dim=1)
        all_true.append(y_user.detach().cpu())
        all_pred.append(pred.detach().cpu())

    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)

    user_acc = compute_accuracy(y_true, y_pred)

    return {
        "loss": total_loss / total_samples,
        "user_acc": user_acc,
        "uia": user_acc,
    }


@torch.no_grad()
def evaluate_user(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.backbone.eval()
    model.task_head.eval()
    model.user_head.eval()

    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    all_true = []
    all_pred = []

    for x, _, y_user, _ in dataloader:
        x = x.to(device)
        y_user = y_user.to(device)

        user_logits = model(x, head="user")
        loss = ce(user_logits, y_user)

        bs = x.size(0)
        total_samples += bs
        total_loss += loss.item() * bs

        pred = user_logits.argmax(dim=1)
        all_true.append(y_user.detach().cpu())
        all_pred.append(pred.detach().cpu())

    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)

    user_acc = compute_accuracy(y_true, y_pred)

    return {
        "loss": total_loss / total_samples,
        "user_acc": user_acc,
        "uia": user_acc,
    }
def build_model_config(ds: Dataset, model_name: str):
    model_name = model_name.strip()

    if model_name == "EEGNet":
        return EEGNetConfig(
            n_channels=ds.n_channels,
            n_times=ds.n_times,
        )

    if model_name == "ShallowCNN":
        return ShallowCNNConfig(
            n_channels=ds.n_channels,
            n_times=ds.n_times,
        )

    raise ValueError(f"Unknown model={model_name!r}")

# =========================
# 5) Main
# =========================
def main():
    
    parser = argparse.ArgumentParser(
        description="Two-stage baseline with LOSO: "
                    "(1) train task model, (2) train user classifier on fixed backbone"
    )

    # ---- dataset / model selection ----
    parser.add_argument(
        "--dataset",
        type=str,
        default="MI1",
        help="Dataset name, e.g. MI1",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EEGNet",
        choices=["EEGNet", "ShallowCNN"],
        help="Model backbone to use",
    )


    # ---- save path ----
    parser.add_argument("--save_root", type=str, default="checkpoint/checkpoints_2stage")
    parser.add_argument("--run_name", type=str, default=None)

    # ---- training ----
    parser.add_argument("--task_epochs", type=int, default=100)
    parser.add_argument("--user_epochs", type=int, default=100)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--task_lr", type=float, default=1e-3)
    parser.add_argument("--user_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--user_hidden_dim", type=int, default=256)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--user_dropout", type=float, default=0.5)

    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Five random seeds, e.g. 0,1,2,3,4",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="none",
        choices=["none", "trial", "channel"],
    )

    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) != 5:
        print(f"[Warn] Paper says repeat 5 times, current seeds={seeds}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- dataset ----
    args.data_dir = "data/" + args.dataset
    ds = build_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        normalize=args.normalize,
    )
    model_cfg_for_run = build_model_config(ds, args.model)
    model_cfg_dict = asdict(model_cfg_for_run)
    user_head_cfg = {
    "hidden_dim": args.user_hidden_dim,
    "dropout": args.user_dropout,
}
    print("\n=== Dataset Summary ===")
    print(f"dataset          : {args.dataset}")
    print(f"model            : {args.model}")
    print(f"data_dir         : {args.data_dir}")
    print(f"n_samples        : {ds.n_samples}")
    print(f"n_channels       : {ds.n_channels}")
    print(f"n_times          : {ds.n_times}")
    print(f"n_task_classes   : {ds.n_task_classes}")
    print(f"n_users          : {ds.n_users}")
    print(f"n_sessions       : {ds.n_sessions}")
    print(f"task labels      : {torch.unique(ds.y_task).tolist()}")
    print(f"user labels      : {torch.unique(ds.y_user).tolist()}")
    print(f"session labels   : {torch.unique(ds.y_session).tolist()}")
    print(f"session originals: {ds.session_original_values}")
    print()

    # ---- save dir ----
    run_dir = make_run_dir(args.save_root + "_"+args.model, args.run_name) 
    print(f"Run directory     : {run_dir}")
    print(f"Device            : {device}")
    print()

    config_to_save = {
        "dataset": args.dataset,
        "model": args.model,
        "data_dir": args.data_dir,
        "save_root": args.save_root,
        "run_name": run_dir.name,
        "task_epochs": args.task_epochs,
        "user_epochs": args.user_epochs,
        "batch_size": args.batch_size,
        "task_lr": args.task_lr,
        "user_lr": args.user_lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "user_hidden_dim": args.user_hidden_dim,
        "save_every": args.save_every,
        "seeds": seeds,
        "normalize": args.normalize,
        "model_config": model_cfg_dict,
        "user_head_config": user_head_cfg,

        "dataset_summary": {
            "n_samples": ds.n_samples,
            "n_channels": ds.n_channels,
            "n_times": ds.n_times,
            "n_task_classes": ds.n_task_classes,
            "n_users": ds.n_users,
            "n_sessions": ds.n_sessions,
            "session_original_values": ds.session_original_values,
        },
    }
    save_json(config_to_save, run_dir / "config.json")

    all_fold_results = []

    # =====================================================
    # Repeat over seeds
    # =====================================================
    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"Seed = {seed}")
        print("=" * 80)
        set_seed(seed)

        # -------------------------------------------------
        # LOSO over sessions
        # -------------------------------------------------
        for train_session_internal in range(ds.n_sessions):
            train_session_original = ds.session_original_values[train_session_internal]

            fold_name = f"seed_{seed}_train_session_{train_session_original}"
            fold_dir = run_dir / fold_name
            fold_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "-" * 80)
            print(
                f"Fold: seed={seed}, "
                f"train_session_internal={train_session_internal}, "
                f"train_session_original={train_session_original}"
            )
            print("-" * 80)

            train_idx, test_idx = build_loso_split(ds, train_session_internal)

            train_set = Subset(ds, train_idx)
            test_set = Subset(ds, test_idx)

            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
            test_loader = DataLoader(
                test_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )

            print(f"train samples: {len(train_set)}")
            print(f"test samples : {len(test_set)}")

            # =====================================================
            # Stage 1: train task model
            # =====================================================
            model, cfg = build_model(
                ds,
                user_hidden_dim=args.user_hidden_dim,
                model_name=args.model,
            )
            model = model.to(device)

            # stage1: only backbone + task_head
            freeze_module(model.user_head)
            unfreeze_module(model.backbone)
            unfreeze_module(model.task_head)

            task_optimizer = torch.optim.Adam(
                list(model.backbone.parameters()) + list(model.task_head.parameters()),
                lr=args.task_lr,
                weight_decay=args.weight_decay,
            )

            best_task_bca = -1.0
            best_task_epoch = -1
            best_task_row = None
            best_task_state = None
            task_history = []

            print("\n[Stage 1] Train task model")
            for epoch in range(1, args.task_epochs + 1):
                train_task_stats = train_task_one_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=task_optimizer,
                    device=device,
                )

                test_task_stats = evaluate_task(
                    model=model,
                    dataloader=test_loader,
                    device=device,
                    n_task_classes=ds.n_task_classes,
                )

                row = {
                    "dataset": args.dataset,
                    "model": args.model,
                    "seed": seed,
                    "train_session_internal": train_session_internal,
                    "train_session_original": train_session_original,
                    "stage": "task",
                    "epoch": epoch,
                    "train_task_loss": train_task_stats["loss"],
                    "train_mi_acc": train_task_stats["mi_acc"],
                    "test_task_loss": test_task_stats["loss"],
                    "test_mi_acc": test_task_stats["mi_acc"],
                    "test_bca": test_task_stats["bca"],
                }
                task_history.append(row)

                print(
                    f"[Task] Epoch {epoch:03d} | "
                    f"train_loss={row['train_task_loss']:.4f} | "
                    f"train_MI_ACC={row['train_mi_acc'] * 100:.2f}% | "
                    f"test_loss={row['test_task_loss']:.4f} | "
                    f"test_MI_ACC={row['test_mi_acc'] * 100:.2f}% | "
                    f"test_BCA={row['test_bca'] * 100:.2f}%"
                )

                extra = {
                    "stage": "task",
                    "dataset_type": args.dataset,
                    "model_name": args.model,
                    "seed": seed,
                    "train_session_internal": train_session_internal,
                    "train_session_original": train_session_original,
                }

                save_checkpoint(
                    path=fold_dir / "latest_task.pth",
                    model=model,
                    optimizer=task_optimizer,
                    epoch=epoch,
                    metrics=row,
                    cfg=cfg,
                    extra=extra,
                )

                if args.save_every > 0 and epoch % args.save_every == 0:
                    save_checkpoint(
                        path=fold_dir / f"task_epoch_{epoch:03d}.pth",
                        model=model,
                        optimizer=task_optimizer,
                        epoch=epoch,
                        metrics=row,
                        cfg=cfg,
                        extra=extra,
                    )

                if row["test_bca"] > best_task_bca:
                    best_task_bca = row["test_bca"]
                    best_task_epoch = epoch
                    best_task_row = dict(row)
                    best_task_state = copy.deepcopy(model.state_dict())

                    save_checkpoint(
                        path=fold_dir / "best_task_by_bca.pth",
                        model=model,
                        optimizer=task_optimizer,
                        epoch=epoch,
                        metrics=row,
                        cfg=cfg,
                        extra=extra,
                    )
                    print("  -> saved best_task_by_bca.pth")

                save_json({"task_history": task_history}, fold_dir / "task_history.json")

            if best_task_state is None:
                raise RuntimeError("best_task_state is None after Stage 1")

            # load best task model before stage 2
            model.load_state_dict(best_task_state)

            # =====================================================
            # Stage 2: train user classifier on fixed backbone
            # =====================================================
            print("\n[Stage 2] Train user classifier on fixed backbone")

            # reset user head to avoid contamination from stage 1
            reset_user_head(
                model=model,
                n_users=ds.n_users,
                hidden_dim=args.user_hidden_dim,
                device=device,
                dropout=args.user_dropout,
            )

            freeze_module(model.backbone)
            freeze_module(model.task_head)
            unfreeze_module(model.user_head)

            user_optimizer = torch.optim.Adam(
                model.user_head.parameters(),
                lr=args.user_lr,
                weight_decay=args.weight_decay,
            )

            best_user_acc = -1.0
            best_user_epoch = -1
            best_user_row = None
            user_history = []

            for epoch in range(1, args.user_epochs + 1):
                train_user_stats = train_user_one_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=user_optimizer,
                    device=device,
                )

                test_user_stats = evaluate_user(
                    model=model,
                    dataloader=test_loader,
                    device=device,
                )

                row = {
                    "dataset": args.dataset,
                    "model": args.model,
                    "seed": seed,
                    "train_session_internal": train_session_internal,
                    "train_session_original": train_session_original,
                    "stage": "user",
                    "epoch": epoch,
                    "train_user_loss": train_user_stats["loss"],
                    "train_user_acc": train_user_stats["user_acc"],
                    "train_uia": train_user_stats["uia"],
                    "test_user_loss": test_user_stats["loss"],
                    "test_user_acc": test_user_stats["user_acc"],
                    "test_uia": test_user_stats["uia"],
                }
                user_history.append(row)

                print(
                    f"[User] Epoch {epoch:03d} | "
                    f"train_loss={row['train_user_loss']:.4f} | "
                    f"train_USER_ACC={row['train_user_acc'] * 100:.2f}% | "
                    f"test_loss={row['test_user_loss']:.4f} | "
                    f"test_USER_ACC={row['test_user_acc'] * 100:.2f}% | "
                    f"test_UIA={row['test_uia'] * 100:.2f}%"
                )

                extra = {
                    "stage": "user",
                    "dataset_type": args.dataset,
                    "model_name": args.model,
                    "seed": seed,
                    "train_session_internal": train_session_internal,
                    "train_session_original": train_session_original,
                    "best_task_epoch": best_task_epoch,
                    "best_task_bca": best_task_bca,
                }

                save_checkpoint(
                    path=fold_dir / "latest_user.pth",
                    model=model,
                    optimizer=user_optimizer,
                    epoch=epoch,
                    metrics=row,
                    cfg=cfg,
                    extra=extra,
                )

                if args.save_every > 0 and epoch % args.save_every == 0:
                    save_checkpoint(
                        path=fold_dir / f"user_epoch_{epoch:03d}.pth",
                        model=model,
                        optimizer=user_optimizer,
                        epoch=epoch,
                        metrics=row,
                        cfg=cfg,
                        extra=extra,
                    )

                if row["test_user_acc"] > best_user_acc:
                    best_user_acc = row["test_user_acc"]
                    best_user_epoch = epoch
                    best_user_row = dict(row)

                    save_checkpoint(
                        path=fold_dir / "best_user_by_acc.pth",
                        model=model,
                        optimizer=user_optimizer,
                        epoch=epoch,
                        metrics=row,
                        cfg=cfg,
                        extra=extra,
                    )
                    print("  -> saved best_user_by_acc.pth")

                save_json({"user_history": user_history}, fold_dir / "user_history.json")

            if best_task_row is None or best_user_row is None:
                raise RuntimeError("best_task_row or best_user_row is None")

            # =====================================================
            # Fold summary
            # =====================================================
            fold_result = {
                "dataset": args.dataset,
                "model": args.model,
                "seed": seed,
                "train_session_internal": train_session_internal,
                "train_session_original": train_session_original,
                "best_task_epoch": best_task_epoch,
                "best_user_epoch": best_user_epoch,
                "test_mi_acc": best_task_row["test_mi_acc"],
                "test_bca": best_task_row["test_bca"],
                "test_user_acc": best_user_row["test_user_acc"],
                "test_uia": best_user_row["test_uia"],
                "task_stage_best": best_task_row,
                "user_stage_best": best_user_row,
            }
            all_fold_results.append(fold_result)

            save_json(
                {
                    "fold_result": fold_result,
                    "task_stage_best": best_task_row,
                    "user_stage_best": best_user_row,
                },
                fold_dir / "fold_summary.json",
            )

    # =====================================================
    # Final summary
    # =====================================================
    def mean_metric(rows, key):
        return float(np.mean([r[key] for r in rows]))

    final_summary = {
        "dataset": args.dataset,
        "model": args.model,
        "num_total_runs": len(all_fold_results),
        "mean_test_mi_acc": mean_metric(all_fold_results, "test_mi_acc"),
        "mean_test_bca": mean_metric(all_fold_results, "test_bca"),
        "mean_test_user_acc": mean_metric(all_fold_results, "test_user_acc"),
        "mean_test_uia": mean_metric(all_fold_results, "test_uia"),
        "all_fold_results": all_fold_results,
    }

    save_json(final_summary, run_dir / "final_summary.json")

    print("\n" + "=" * 80)
    print("Training finished.")
    print(f"Dataset: {args.dataset}")
    print(f"Model  : {args.model}")
    print(f"Run dir: {run_dir}")
    print(f"Total runs: {len(all_fold_results)}")
    print()
    print(f"Mean MI_ACC   : {final_summary['mean_test_mi_acc'] * 100:.2f}%")
    print(f"Mean BCA      : {final_summary['mean_test_bca'] * 100:.2f}%")
    print(f"Mean USER_ACC : {final_summary['mean_test_user_acc'] * 100:.2f}%")
    print(f"Mean UIA      : {final_summary['mean_test_uia'] * 100:.2f}%")
    print("\nSaved:")
    print(f"  - {run_dir / 'config.json'}")
    print(f"  - {run_dir / 'final_summary.json'}")


if __name__ == "__main__":
    main()
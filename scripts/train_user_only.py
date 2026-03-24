from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import scipy.io as scio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from eegnet import EEGNetConfig, EEGNetFeatureExtractor, UserClassifier


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
        y_user_all = []
        y_session_all = []

        print("Loaded mat files:")
        for f in mat_files:
            print(" ", f.name)

        for mat_idx, mat_path in enumerate(mat_files):
            data = scio.loadmat(mat_path)

            if "X" not in data or "session" not in data:
                raise ValueError(f"Missing X/session in {mat_path}")

            X = np.asarray(data["X"], dtype=np.float32)           # (N, C, T)
            y_session = np.asarray(data["session"]).reshape(-1)

            # session -> int
            y_session = np.array([int(v) for v in y_session], dtype=np.int64)

            if X.shape[0] != len(y_session):
                raise ValueError(
                    f"Mismatch in {mat_path}: X.shape[0]={X.shape[0]}, len(session)={len(y_session)}"
                )

            # ---------- optional normalization ----------
            if normalize == "trial":
                # each trial over all channels+times
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
            y_user_all.append(np.full(X.shape[0], mat_idx, dtype=np.int64))
            y_session_all.append(y_session)

        X_all = np.concatenate(X_all, axis=0)
        y_user_all = np.concatenate(y_user_all, axis=0)
        y_session_all = np.concatenate(y_session_all, axis=0)

        uniq_session = sorted(np.unique(y_session_all).tolist())
        session_map = {v: i for i, v in enumerate(uniq_session)}
        y_session_internal = np.array([session_map[v] for v in y_session_all], dtype=np.int64)

        self.x = torch.tensor(X_all, dtype=torch.float32)
        self.y_user = torch.tensor(y_user_all, dtype=torch.long)
        self.y_session = torch.tensor(y_session_internal, dtype=torch.long)

        self.session_original_values = uniq_session
        self.session_map = session_map

        self.n_samples = self.x.shape[0]
        self.n_channels = self.x.shape[1]
        self.n_times = self.x.shape[2]
        self.n_users = int(torch.unique(self.y_user).numel())
        self.n_sessions = int(torch.unique(self.y_session).numel())

        print("\nDataset loaded:")
        print("samples:", self.n_samples)
        print("channels:", self.n_channels)
        print("times:", self.n_times)
        print("users:", self.n_users)
        print("sessions:", self.n_sessions)
        print("original session values:", self.session_original_values)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y_user[idx], self.y_session[idx]


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


def build_loso_split(ds: MI1Dataset, train_session_internal: int) -> Tuple[List[int], List[int]]:
    all_sessions = ds.y_session.numpy()
    train_idx = np.where(all_sessions == train_session_internal)[0].tolist()
    test_idx = np.where(all_sessions != train_session_internal)[0].tolist()

    if len(train_idx) == 0:
        raise ValueError(f"No samples for train_session_internal={train_session_internal}")
    if len(test_idx) == 0:
        raise ValueError(f"No samples left for test when training on session {train_session_internal}")

    return train_idx, test_idx


def compute_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
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


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: Dict,
    cfg: EEGNetConfig,
    extra: Dict,
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "metrics": metrics,
        "eegnet_config": asdict(cfg),
        "extra": extra,
    }
    torch.save(ckpt, path)


# =========================
# 3) Model: user-only
# =========================
class EEGNetUserOnly(nn.Module):
    """
    backbone + user classifier only
    input  : (B, C, T)
    output : (B, n_users)
    """
    def __init__(self, cfg: EEGNetConfig, n_users: int, user_hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.backbone = EEGNetFeatureExtractor(cfg)
        self.user_head = UserClassifier(
            feature_dim=self.backbone.feature_dim,
            n_users=n_users,
            hidden_dim=user_hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.backbone(x)
        logits = self.user_head(feat)
        return (logits, feat) if return_features else logits


def build_model(ds: MI1Dataset, user_hidden_dim: int, user_dropout: float) -> Tuple[EEGNetUserOnly, EEGNetConfig]:
    cfg = EEGNetConfig(
        n_channels=ds.n_channels,
        n_times=ds.n_times,
    )
    model = EEGNetUserOnly(
        cfg=cfg,
        n_users=ds.n_users,
        user_hidden_dim=user_hidden_dim,
        dropout=user_dropout,
    )
    return model, cfg


# =========================
# 4) Train / Eval
# =========================
def train_one_epoch(
    model: EEGNetUserOnly,
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

    for x, y_user, _ in dataloader:
        x = x.to(device)
        y_user = y_user.to(device)

        optimizer.zero_grad()

        user_logits = model(x)
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
    user_bca = compute_bca(y_true, y_pred, n_classes=model.user_head.fc2.out_features)

    return {
        "loss": total_loss / total_samples,
        "user_acc": user_acc,
        "uia": user_acc,
        "user_bca": user_bca,
    }


@torch.no_grad()
def evaluate(
    model: EEGNetUserOnly,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    all_true = []
    all_pred = []

    for x, y_user, _ in dataloader:
        x = x.to(device)
        y_user = y_user.to(device)

        user_logits = model(x)
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
    user_bca = compute_bca(y_true, y_pred, n_classes=model.user_head.fc2.out_features)

    return {
        "loss": total_loss / total_samples,
        "user_acc": user_acc,
        "uia": user_acc,
        "user_bca": user_bca,
    }


# =========================
# 5) Main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="User-only EEGNet baseline on MI1 with LOSO session split"
    )
    parser.add_argument("--mi1_dir", type=str, default="data/MI1")
    parser.add_argument("--save_root", type=str, default="checkpoint/checkpoints_MI1_user_only")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--user_hidden_dim", type=int, default=128)
    parser.add_argument("--user_dropout", type=float, default=0.5)
    parser.add_argument("--normalize", type=str, default="channel", choices=["none", "trial", "channel"])
    parser.add_argument("--save_every", type=int, default=10)

    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Five random seeds, e.g. 0,1,2,3,4",
    )
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) != 5:
        print(f"[Warn] Paper says repeat 5 times, current seeds={seeds}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MI1Dataset(args.mi1_dir, normalize=args.normalize)

    print("\n=== Dataset Summary ===")
    print(f"mi1_dir          : {args.mi1_dir}")
    print(f"n_samples        : {ds.n_samples}")
    print(f"n_channels       : {ds.n_channels}")
    print(f"n_times          : {ds.n_times}")
    print(f"n_users          : {ds.n_users}")
    print(f"n_sessions       : {ds.n_sessions}")
    print(f"user labels      : {torch.unique(ds.y_user).tolist()}")
    print(f"session labels   : {torch.unique(ds.y_session).tolist()}")
    print(f"session originals: {ds.session_original_values}")
    print()

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
        "num_workers": args.num_workers,
        "user_hidden_dim": args.user_hidden_dim,
        "user_dropout": args.user_dropout,
        "normalize": args.normalize,
        "save_every": args.save_every,
        "seeds": seeds,
        "dataset_summary": {
            "n_samples": ds.n_samples,
            "n_channels": ds.n_channels,
            "n_times": ds.n_times,
            "n_users": ds.n_users,
            "n_sessions": ds.n_sessions,
            "session_original_values": ds.session_original_values,
        },
    }
    save_json(config_to_save, run_dir / "config.json")

    all_fold_results = []

    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"Seed = {seed}")
        print("=" * 80)
        set_seed(seed)

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

            model, cfg = build_model(
                ds=ds,
                user_hidden_dim=args.user_hidden_dim,
                user_dropout=args.user_dropout,
            )
            model = model.to(device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            best_user_acc = -1.0
            best_epoch = -1
            best_row = None
            best_state = None
            history = []

            for epoch in range(1, args.epochs + 1):
                train_stats = train_one_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    device=device,
                )

                test_stats = evaluate(
                    model=model,
                    dataloader=test_loader,
                    device=device,
                )

                row = {
                    "seed": seed,
                    "train_session_internal": train_session_internal,
                    "train_session_original": train_session_original,
                    "epoch": epoch,
                    "train_loss": train_stats["loss"],
                    "train_user_acc": train_stats["user_acc"],
                    "train_uia": train_stats["uia"],
                    "train_user_bca": train_stats["user_bca"],
                    "test_loss": test_stats["loss"],
                    "test_user_acc": test_stats["user_acc"],
                    "test_uia": test_stats["uia"],
                    "test_user_bca": test_stats["user_bca"],
                }
                history.append(row)

                print(
                    f"[UserOnly] Epoch {epoch:03d} | "
                    f"train_loss={row['train_loss']:.4f} | "
                    f"train_USER_ACC={row['train_user_acc'] * 100:.2f}% | "
                    f"train_USER_BCA={row['train_user_bca'] * 100:.2f}% | "
                    f"test_loss={row['test_loss']:.4f} | "
                    f"test_USER_ACC={row['test_user_acc'] * 100:.2f}% | "
                    f"test_USER_BCA={row['test_user_bca'] * 100:.2f}% | "
                    f"test_UIA={row['test_uia'] * 100:.2f}%"
                )

                extra = {
                    "stage": "user_only",
                    "dataset_type": "MI1",
                    "seed": seed,
                    "train_session_internal": train_session_internal,
                    "train_session_original": train_session_original,
                }

                save_checkpoint(
                    path=fold_dir / "latest_user_only.pth",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=row,
                    cfg=cfg,
                    extra=extra,
                )

                if args.save_every > 0 and epoch % args.save_every == 0:
                    save_checkpoint(
                        path=fold_dir / f"user_only_epoch_{epoch:03d}.pth",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics=row,
                        cfg=cfg,
                        extra=extra,
                    )

                if row["test_user_acc"] > best_user_acc:
                    best_user_acc = row["test_user_acc"]
                    best_epoch = epoch
                    best_row = dict(row)
                    best_state = copy.deepcopy(model.state_dict())

                    save_checkpoint(
                        path=fold_dir / "best_user_only_by_acc.pth",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics=row,
                        cfg=cfg,
                        extra=extra,
                    )
                    print("  -> saved best_user_only_by_acc.pth")

                save_json({"history": history}, fold_dir / "history.json")

            if best_row is None or best_state is None:
                raise RuntimeError("best_row or best_state is None")

            fold_result = {
                "seed": seed,
                "train_session_internal": train_session_internal,
                "train_session_original": train_session_original,
                "best_epoch": best_epoch,
                "test_user_acc": best_row["test_user_acc"],
                "test_uia": best_row["test_uia"],
                "test_user_bca": best_row["test_user_bca"],
                "best_row": best_row,
            }
            all_fold_results.append(fold_result)

            save_json(
                {
                    "fold_result": fold_result,
                    "best_row": best_row,
                },
                fold_dir / "fold_summary.json",
            )

    def mean_metric(rows, key):
        return float(np.mean([r[key] for r in rows]))

    final_summary = {
        "num_total_runs": len(all_fold_results),
        "mean_test_user_acc": mean_metric(all_fold_results, "test_user_acc"),
        "mean_test_uia": mean_metric(all_fold_results, "test_uia"),
        "mean_test_user_bca": mean_metric(all_fold_results, "test_user_bca"),
        "all_fold_results": all_fold_results,
    }

    save_json(final_summary, run_dir / "final_summary.json")

    print("\n" + "=" * 80)
    print("Training finished.")
    print(f"Run dir: {run_dir}")
    print(f"Total runs: {len(all_fold_results)}")
    print()
    print(f"Mean USER_ACC : {final_summary['mean_test_user_acc'] * 100:.2f}%")
    print(f"Mean UIA      : {final_summary['mean_test_uia'] * 100:.2f}%")
    print(f"Mean USER_BCA : {final_summary['mean_test_user_bca'] * 100:.2f}%")
    print("\nSaved:")
    print(f"  - {run_dir / 'config.json'}")
    print(f"  - {run_dir / 'final_summary.json'}")


if __name__ == "__main__":
    main()
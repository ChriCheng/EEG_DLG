from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext

import torch
from torch import nn


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


@dataclass
class EvalResult:
    mi_acc: float
    user_acc: float
    bca: float
    uia: float


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    dataloader,
    *,
    n_task_classes: int,
    device: torch.device,
    amp: bool = False,
) -> EvalResult:
    model.eval()

    all_task_true = []
    all_task_pred = []
    all_user_true = []
    all_user_pred = []

    if amp and device.type == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = nullcontext()

    for batch in dataloader:
        # 兼容两种格式：
        # 1) (x, y_task, y_user)
        # 2) (x, y_task, y_user, y_session)
        if len(batch) == 3:
            x, y_task, y_user = batch
        elif len(batch) == 4:
            x, y_task, y_user, _ = batch
        else:
            raise ValueError(
                "Each batch must be (x, y_task, y_user) "
                "or (x, y_task, y_user, y_session)"
            )

        x = x.to(device)
        y_task = y_task.to(device)
        y_user = y_user.to(device)

        with autocast_ctx:
            out = model(x)
            if not (isinstance(out, (tuple, list)) and len(out) >= 2):
                raise ValueError("model(x) must return (task_logits, user_logits)")
            task_logits, user_logits = out[0], out[1]

        task_pred = task_logits.argmax(dim=1)
        user_pred = user_logits.argmax(dim=1)

        all_task_true.append(y_task.detach().cpu())
        all_task_pred.append(task_pred.detach().cpu())
        all_user_true.append(y_user.detach().cpu())
        all_user_pred.append(user_pred.detach().cpu())

    if len(all_task_true) == 0:
        return EvalResult(
            mi_acc=0.0,
            user_acc=0.0,
            bca=0.0,
            uia=0.0,
        )

    y_task_true = torch.cat(all_task_true, dim=0)
    y_task_pred = torch.cat(all_task_pred, dim=0)
    y_user_true = torch.cat(all_user_true, dim=0)
    y_user_pred = torch.cat(all_user_pred, dim=0)

    mi_acc = compute_accuracy(y_task_true, y_task_pred)
    user_acc = compute_accuracy(y_user_true, y_user_pred)
    bca = compute_bca(y_task_true, y_task_pred, n_classes=n_task_classes)

    # 当前先把 UIA 视为 user_acc
    uia = user_acc

    return EvalResult(
        mi_acc=mi_acc,
        user_acc=user_acc,
        bca=bca,
        uia=uia,
    )
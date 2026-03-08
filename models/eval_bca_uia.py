from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple
from contextlib import nullcontext
import torch
from torch import nn


# ----------------------------
# 1) Metrics: BCA & UIA (paper Eq.(6)(7))
# ----------------------------
def compute_bca(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    n_classes: int,
    *,
    ignore_empty_classes: bool = True,
) -> float:
    """
    Balanced Classification Accuracy (BCA):
      BCA = (1/K) * sum_k ( 1/N_k * sum_{i in class k} 1(y_pred_i == k) )
    i.e., mean per-class recall. :contentReference[oaicite:2]{index=2}
    """
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(f"y_true/y_pred must be 1-D, got {y_true.shape}, {y_pred.shape}")
    if y_true.numel() != y_pred.numel():
        raise ValueError("y_true and y_pred must have same length")

    # y_true = y_true.to(torch.long)
    # y_pred = y_pred.to(torch.long)

    correct = (y_true == y_pred).to(torch.long)  # (N,)
    # total per class: N_k
    total_per_class = torch.bincount(y_true, minlength=n_classes).to(torch.float32)
    # correct per class: sum_{i in class k} 1(y_pred_i == k) == sum correct for that class
    correct_per_class = torch.bincount(y_true, weights=correct.to(torch.float32), minlength=n_classes)

    if ignore_empty_classes:
        mask = total_per_class > 0
        if mask.sum().item() == 0:
            return 0.0
        per_class_recall = correct_per_class[mask] / total_per_class[mask]
        return per_class_recall.mean().item()
    else:
        # empty classes contribute 0/0 -> set to 0
        per_class_recall = torch.zeros_like(total_per_class)
        nonzero = total_per_class > 0
        per_class_recall[nonzero] = correct_per_class[nonzero] / total_per_class[nonzero]
        return per_class_recall.mean().item()


def compute_uia(y_true_user: torch.Tensor, y_pred_user: torch.Tensor) -> float:
    """
    User Identification Accuracy (UIA):
      UIA = (1/N_t) * sum_i 1(u_pred_i == u_i) :contentReference[oaicite:3]{index=3}
    """
    if y_true_user.ndim != 1 or y_pred_user.ndim != 1:
        raise ValueError(
            f"y_true_user/y_pred_user must be 1-D, got {y_true_user.shape}, {y_pred_user.shape}"
        )
    if y_true_user.numel() == 0:
        return 0.0
    return (y_true_user.to(torch.long) == y_pred_user.to(torch.long)).float().mean().item()


# ----------------------------
# 2) A small evaluation runner (model-agnostic)
# ----------------------------
@dataclass
class EvalResult:
    bca: float
    uia: float


@torch.no_grad()
def evaluate_bca_uia(
    model: nn.Module,
    dataloader,
    *,
    n_task_classes: int,
    device: torch.device,
    amp: bool = False,
) -> EvalResult:
    """
    Assumes each batch yields:
      x: (B, C, T)
      y_task: (B,) long
      y_user: (B,) long
    And model(x) returns (task_logits, user_logits) with shapes:
      task_logits: (B, n_task_classes)
      user_logits: (B, n_users)
    """
    model.eval()

    all_task_true = []
    all_task_pred = []
    all_user_true = []
    all_user_pred = []

    autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16) if amp else nullcontext()

    for batch in dataloader:
        if len(batch) != 3:
            raise ValueError("Each batch must be (x, y_task, y_user)")
        x, y_task, y_user = batch
        x = x.to(device)
        y_task = y_task.to(device)
        y_user = y_user.to(device)

        with autocast_ctx:
            out = model(x)
            if not (isinstance(out, (tuple, list)) and len(out) >= 2):
                raise ValueError("model(x) must return (task_logits, user_logits) or longer tuple")
            task_logits, user_logits = out[0], out[1]

        task_pred = task_logits.argmax(dim=1)
        user_pred = user_logits.argmax(dim=1)

        all_task_true.append(y_task.detach().cpu())
        all_task_pred.append(task_pred.detach().cpu())
        all_user_true.append(y_user.detach().cpu())
        all_user_pred.append(user_pred.detach().cpu())

    y_task_true = torch.cat(all_task_true, dim=0)
    y_task_pred = torch.cat(all_task_pred, dim=0)
    y_user_true = torch.cat(all_user_true, dim=0)
    y_user_pred = torch.cat(all_user_pred, dim=0)

    bca = compute_bca(y_task_true, y_task_pred, n_classes=n_task_classes)
    uia = compute_uia(y_user_true, y_user_pred)
    return EvalResult(bca=bca, uia=uia)


# ----------------------------
# 3) Extensible model builder registry (EEGNet now; add DeepCNN/ShallowCNN later)
# ----------------------------
ModelBuilder = Callable[..., nn.Module]


def build_eegnet(
    *,
    n_channels: int,
    n_times: int,
    n_task_classes: int,
    n_users: int,
) -> nn.Module:
    """
    Calls your existing eegnet implementation.
    Expected in eegnet.py:
      - EEGNetConfig
      - EEGNetMI1MI2 (recommended) OR EEGNetWithHeads
    """
    from eegnet import EEGNetConfig, EEGNetMI1MI2  # <-- 你的 eegnet 文件

    cfg = EEGNetConfig(n_channels=n_channels, n_times=n_times)
    return EEGNetMI1MI2(cfg, n_task_classes=n_task_classes, n_users=n_users)


MODEL_REGISTRY: Dict[str, ModelBuilder] = {
    "eegnet": build_eegnet,
    # "deepcnn": build_deepcnn,       # 之后加
    # "shallowcnn": build_shallowcnn, # 之后加
}





def main():
    # ---- 你需要按 MI1/MI2 改这里的参数 ----
    model_name = "eegnet"
    n_channels = 22
    n_times = 1000
    n_task_classes = 4
    n_users = 9

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    model = MODEL_REGISTRY[model_name](
        n_channels=n_channels,
        n_times=n_times,
        n_task_classes=n_task_classes,
        n_users=n_users,
    ).to(device)

    # demo dataloader (换成你自己的 MI1/MI2 dataloader)
    
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    # evaluate
    res = evaluate_bca_uia(model, dl, n_task_classes=n_task_classes, device=device, amp=False)
    print(f"BCA = {res.bca * 100:.2f}%, UIA = {res.uia * 100:.2f}%")


if __name__ == "__main__":
    main()
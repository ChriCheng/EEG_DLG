from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

#F1=16, D=4, F2=64, pool1=2, pool2=4 kernel_length: int = 64
@dataclass
class EEGNetConfig:
    n_channels: int
    n_times: int
    F1: int = 12
    D: int = 2
    F2: int = 24
    kernel_length: int = 64
    pool1: int = 2
    pool2: int = 4
    dropout: float = 0.25
    temporal_kernels: tuple[int, ...] =(7,15,31)

class MultiScaleTemporalConv(nn.Module):
    """
    Input:  (B, 1, C, T)
    Output: (B, F1, C, T)
    """
    def __init__(self, out_channels: int, kernels: tuple[int, ...]):
        super().__init__()
        if len(kernels) == 0:
            raise ValueError("kernels must not be empty")

        self.kernels = kernels
        n_branches = len(kernels)

        # 尽量平均分配每个分支的输出通道
        base = out_channels // n_branches
        rem = out_channels % n_branches
        branch_outs = [base + (1 if i < rem else 0) for i in range(n_branches)]

        self.branches = nn.ModuleList()
        for k, ch in zip(kernels, branch_outs):
            if k % 2 == 0:
                raise ValueError(
                    f"Please use odd kernel sizes for easy concat, got k={k}"
                )
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=ch,
                        kernel_size=(1, k),
                        padding=(0, k // 2),
                        bias=False,
                    )
                )
            )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]   # each: (B, ch_i, C, T)
        x = torch.cat(outs, dim=1)                       # (B, F1, C, T)
        x = self.bn(x)
        return x

class EEGNetFeatureExtractor(nn.Module):
    """
    Input:  (B, C, T)
    Output: (B, feature_dim)
    """
    def __init__(self, cfg: EEGNetConfig):
        super().__init__()
        self.cfg = cfg

        # 用多尺度分支替换原来的 first_conv
        self.first_conv = MultiScaleTemporalConv(
            out_channels=cfg.F1,
            kernels=cfg.temporal_kernels,
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg.F1,
                out_channels=cfg.F1 * cfg.D,
                kernel_size=(cfg.n_channels, 1),
                groups=cfg.F1,
                bias=False,
            ),
            nn.BatchNorm2d(cfg.F1 * cfg.D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, cfg.pool1)),
            nn.Dropout(cfg.dropout),
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg.F1 * cfg.D,
                out_channels=cfg.F2,
                kernel_size=(1, 16),
                padding=(0, 16 // 2),
                bias=False,
            ),
            nn.BatchNorm2d(cfg.F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, cfg.pool2)),
            nn.Dropout(cfg.dropout),
        )

        self._feature_dim = self._infer_feature_dim(cfg)

    def _infer_feature_dim(self, cfg: EEGNetConfig) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, cfg.n_channels, cfg.n_times)
            x = self.first_conv(dummy)
            x = self.depthwise_conv(x)
            x = self.separable_conv(x)
            return x.view(1, -1).shape[1]

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B,C,T), got {tuple(x.shape)}")
        x = x.unsqueeze(1)      # (B,1,C,T)
        x = self.first_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        return x.flatten(start_dim=1)
    
# class EEGNetFeatureExtractor(nn.Module):
#     """
#     Input:  (B, C, T)
#     Output: (B, feature_dim)
#     """

#     def __init__(self, cfg: EEGNetConfig):
#         super().__init__()
#         self.cfg = cfg

#         self.first_conv = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=1,
#                 out_channels=cfg.F1,
#                 kernel_size=(1, cfg.kernel_length),
#                 padding=(0, cfg.kernel_length // 2),
#                 bias=False,
#             ),
#             nn.BatchNorm2d(cfg.F1),
#         )

#         self.depthwise_conv = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=cfg.F1,
#                 out_channels=cfg.F1 * cfg.D,
#                 kernel_size=(cfg.n_channels, 1),
#                 groups=cfg.F1,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(cfg.F1 * cfg.D),
#             nn.ELU(),
#             nn.AvgPool2d(kernel_size=(1, cfg.pool1)),
#             nn.Dropout(cfg.dropout),
#         )

#         self.separable_conv = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=cfg.F1 * cfg.D,
#                 out_channels=cfg.F2,
#                 kernel_size=(1, 16),
#                 padding=(0, 16 // 2),
#                 bias=False,
#             ),
#             nn.BatchNorm2d(cfg.F2),
#             nn.ELU(),
#             nn.AvgPool2d(kernel_size=(1, cfg.pool2)),
#             nn.Dropout(cfg.dropout),
#         )

#         self._feature_dim = self._infer_feature_dim(cfg)

#     @staticmethod
#     def _build_conv_only(cfg: EEGNetConfig) -> nn.Sequential:
#         return nn.Sequential(
#             nn.Sequential(
#                 nn.Conv2d(
#                     in_channels=1,
#                     out_channels=cfg.F1,
#                     kernel_size=(1, cfg.kernel_length),
#                     padding=(0, cfg.kernel_length // 2),
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(cfg.F1),
#             ),
#             nn.Sequential(
#                 nn.Conv2d(
#                     in_channels=cfg.F1,
#                     out_channels=cfg.F1 * cfg.D,
#                     kernel_size=(cfg.n_channels, 1),
#                     groups=cfg.F1,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(cfg.F1 * cfg.D),
#                 nn.ELU(),
#                 nn.AvgPool2d(kernel_size=(1, cfg.pool1)),
#                 nn.Dropout(cfg.dropout),
#             ),
#             nn.Sequential(
#                 nn.Conv2d(
#                     in_channels=cfg.F1 * cfg.D,
#                     out_channels=cfg.F2,
#                     kernel_size=(1, 16),
#                     padding=(0, 16 // 2),
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(cfg.F2),
#                 nn.ELU(),
#                 nn.AvgPool2d(kernel_size=(1, cfg.pool2)),
#                 nn.Dropout(cfg.dropout),
#             ),
#         )

#     @staticmethod
#     def _infer_feature_dim(cfg: EEGNetConfig) -> int:
#         with torch.no_grad():
#             dummy = torch.zeros(1, 1, cfg.n_channels, cfg.n_times)
#             conv = EEGNetFeatureExtractor._build_conv_only(cfg)
#             out = conv(dummy)
#             return out.view(1, -1).shape[1]

#     @property
#     def feature_dim(self) -> int:
#         return self._feature_dim

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.ndim != 3:
#             raise ValueError(f"Expected (B,C,T), got {tuple(x.shape)}")
#         x = x.unsqueeze(1)      # (B,1,C,T)
#         x = self.first_conv(x)
#         x = self.depthwise_conv(x)
#         x = self.separable_conv(x)
#         return x.flatten(start_dim=1)  # (B, feature_dim)


class TaskClassifier(nn.Module):
    """论文：one fully-connected layer as Task-Classifier"""
    def __init__(self, feature_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, n_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat)


class UserClassifier(nn.Module):
    """论文：two fully-connected layers as User-Classifier"""
    def __init__(self, feature_dim: int, n_users: int, hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.act = nn.ELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_users)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.fc1(feat)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


class EEGNetMI1MI2(nn.Module):
    """
    EEGNet feature extractor + Task head + User head
    - task_logits: (B, n_task_classes)  # MI 分类(例如 left/right/foot/tongue 或者二分类)
    - user_logits: (B, n_users)         # subject id 分类（MI1/MI2 通常就是受试者ID数）
    """

    def __init__(self, cfg: EEGNetConfig, n_task_classes: int, n_users: int, user_hidden_dim: int = 128):
        super().__init__()
        self.backbone = EEGNetFeatureExtractor(cfg)
        self.task_head = TaskClassifier(self.backbone.feature_dim, n_task_classes)
        self.user_head = UserClassifier(self.backbone.feature_dim, n_users, hidden_dim=user_hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_features: bool = False,
        head: str = "both",  # "task" | "user" | "both"
    ):
        feat = self.backbone(x)
        if head == "task":
            out = self.task_head(feat)
            return (out, feat) if return_features else out
        if head == "user":
            out = self.user_head(feat)
            return (out, feat) if return_features else out
        if head == "both":
            task_logits = self.task_head(feat)
            user_logits = self.user_head(feat)
            return (task_logits, user_logits, feat) if return_features else (task_logits, user_logits)
        raise ValueError(f"Unknown head={head!r}")


if __name__ == "__main__":
    cfg = EEGNetConfig(n_channels=22, n_times=1000)
    model = EEGNetMI1MI2(cfg, n_task_classes=4, n_users=9, user_hidden_dim=128)

    x = torch.randn(8, cfg.n_channels, cfg.n_times)
    task_logits, user_logits, feat = model(x, return_features=True, head="both")
    print(task_logits.shape, user_logits.shape, feat.shape)
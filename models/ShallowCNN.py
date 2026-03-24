from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class ShallowCNNConfig:
    n_channels: int
    n_times: int

    # classic shallow convnet / shallowfbcsp-like defaults
    n_filters_time: int = 40
    filter_time_length: int = 25
    n_filters_spat: int = 40
    pool_time_length: int = 75
    pool_time_stride: int = 15
    dropout: float = 0.5
    batch_norm: bool = True
    log_eps: float = 1e-6


class Square(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class SafeLog(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=self.eps))


class ShallowCNNFeatureExtractor(nn.Module):
    """
    Input : (B, C, T)
    Output: (B, feature_dim)

    Architecture:
      temporal conv -> spatial conv -> BN -> square -> avgpool -> log -> dropout -> flatten
    """

    def __init__(self, cfg: ShallowCNNConfig):
        super().__init__()
        self.cfg = cfg

        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=cfg.n_filters_time,
            kernel_size=(1, cfg.filter_time_length),
            stride=1,
            padding=0,
            bias=True,
        )

        self.spatial_conv = nn.Conv2d(
            in_channels=cfg.n_filters_time,
            out_channels=cfg.n_filters_spat,
            kernel_size=(cfg.n_channels, 1),
            stride=1,
            padding=0,
            bias=not cfg.batch_norm,
        )

        self.bn = nn.BatchNorm2d(cfg.n_filters_spat) if cfg.batch_norm else nn.Identity()
        self.square = Square()
        self.pool = nn.AvgPool2d(
            kernel_size=(1, cfg.pool_time_length),
            stride=(1, cfg.pool_time_stride),
        )
        self.log = SafeLog(cfg.log_eps)
        self.drop = nn.Dropout(cfg.dropout)

        self._feature_dim = self._infer_feature_dim(cfg)

    @staticmethod
    def _build_conv_only(cfg: ShallowCNNConfig) -> nn.Sequential:
        bn = nn.BatchNorm2d(cfg.n_filters_spat) if cfg.batch_norm else nn.Identity()
        return nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=cfg.n_filters_time,
                kernel_size=(1, cfg.filter_time_length),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Conv2d(
                in_channels=cfg.n_filters_time,
                out_channels=cfg.n_filters_spat,
                kernel_size=(cfg.n_channels, 1),
                stride=1,
                padding=0,
                bias=not cfg.batch_norm,
            ),
            bn,
            Square(),
            nn.AvgPool2d(
                kernel_size=(1, cfg.pool_time_length),
                stride=(1, cfg.pool_time_stride),
            ),
            SafeLog(cfg.log_eps),
            nn.Dropout(cfg.dropout),
        )

    @staticmethod
    def _infer_feature_dim(cfg: ShallowCNNConfig) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, cfg.n_channels, cfg.n_times)
            conv = ShallowCNNFeatureExtractor._build_conv_only(cfg)
            out = conv(dummy)
            if out.numel() == 0:
                raise ValueError(
                    "ShallowCNN produced empty output. "
                    f"Please check n_times={cfg.n_times}, "
                    f"filter_time_length={cfg.filter_time_length}, "
                    f"pool_time_length={cfg.pool_time_length}, "
                    f"pool_time_stride={cfg.pool_time_stride}."
                )
            return out.view(1, -1).shape[1]

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B,C,T), got {tuple(x.shape)}")

        x = x.unsqueeze(1)  # (B,1,C,T)
        x = self.temporal_conv(x)   # (B, Ft, C, T')
        x = self.spatial_conv(x)    # (B, Fs, 1, T')
        x = self.bn(x)
        x = self.square(x)
        x = self.pool(x)
        x = self.log(x)
        x = self.drop(x)
        return x.flatten(start_dim=1)


class TaskClassifier(nn.Module):
    """one fully-connected layer as Task-Classifier"""
    def __init__(self, feature_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, n_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat)


class UserClassifier(nn.Module):
    """two fully-connected layers as User-Classifier"""
    def __init__(
        self,
        feature_dim: int,
        n_users: int,
        hidden_dim: int = 128,
        dropout: float = 0.5,
    ):
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


class ShallowCNNMI1MI2(nn.Module):
    """
    ShallowCNN feature extractor + Task head + User head
    """

    def __init__(
        self,
        cfg: ShallowCNNConfig,
        n_task_classes: int,
        n_users: int,
        user_hidden_dim: int = 128,
        user_dropout: float = 0.5,
    ):
        super().__init__()
        self.backbone = ShallowCNNFeatureExtractor(cfg)
        self.task_head = TaskClassifier(self.backbone.feature_dim, n_task_classes)
        self.user_head = UserClassifier(
            self.backbone.feature_dim,
            n_users,
            hidden_dim=user_hidden_dim,
            dropout=user_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_features: bool = False,
        head: str = "both",   # "task" | "user" | "both"
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


class ShallowCNNUserOnly(nn.Module):
    """
    backbone + user classifier only
    """

    def __init__(
        self,
        cfg: ShallowCNNConfig,
        n_users: int,
        user_hidden_dim: int = 128,
        user_dropout: float = 0.5,
    ):
        super().__init__()
        self.backbone = ShallowCNNFeatureExtractor(cfg)
        self.user_head = UserClassifier(
            feature_dim=self.backbone.feature_dim,
            n_users=n_users,
            hidden_dim=user_hidden_dim,
            dropout=user_dropout,
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.backbone(x)
        logits = self.user_head(feat)
        return (logits, feat) if return_features else logits


if __name__ == "__main__":
    cfg = ShallowCNNConfig(n_channels=22, n_times=1000)

    model = ShallowCNNMI1MI2(
        cfg=cfg,
        n_task_classes=4,
        n_users=9,
        user_hidden_dim=128,
        user_dropout=0.5,
    )

    x = torch.randn(8, cfg.n_channels, cfg.n_times)
    task_logits, user_logits, feat = model(x, return_features=True, head="both")
    print("task_logits:", task_logits.shape)
    print("user_logits:", user_logits.shape)
    print("feat:", feat.shape)
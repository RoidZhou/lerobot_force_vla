#!/usr/bin/env python

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class Tac3DGridMixin:
    input_shape: tuple[int, int]
    input_channels: int

    def _as_grid(self, x: Tensor) -> Tensor:
        if x.ndim == 4 and x.shape[1] == self.input_channels:
            return x
        if x.ndim == 3:
            bsize, n_taxels, channels = x.shape
            expected_taxels = self.input_shape[0] * self.input_shape[1]
            if n_taxels != expected_taxels or channels != self.input_channels:
                raise ValueError(
                    f"Tac3D input expected (B, {expected_taxels}, {self.input_channels}), got {x.shape}."
                )
            return x.reshape(bsize, self.input_shape[0], self.input_shape[1], channels).permute(0, 3, 1, 2)
        raise ValueError(f"Tac3D input expected (B, N, C) or (B, C, H, W), got {x.shape}.")


class Tac3DCNN(Tac3DGridMixin, nn.Module):
    """Encode Tac3D taxel vector fields shaped as (B, 400, 3) or (B, 3, 20, 20)."""

    def __init__(
        self,
        input_shape: tuple[int, int] = (20, 20),
        input_channels: int = 3,
        feature_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.input_channels = input_channels

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(128, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._as_grid(x)
        x = self.pool(F.silu(self.bn1(self.conv1(x))))
        x = self.pool(F.silu(self.bn2(self.conv2(x))))
        x = F.silu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).flatten(1)
        return self.proj(x)


class Tac3DAttentionCNN(Tac3DGridMixin, nn.Module):
    """Tac3D CNN with spatial attention over the taxel grid."""

    def __init__(
        self,
        input_shape: tuple[int, int] = (20, 20),
        input_channels: int = 3,
        feature_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.input_channels = input_channels

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._as_grid(x)
        x = self.pool(F.silu(self.bn1(self.conv1(x))))
        x = self.pool(F.silu(self.bn2(self.conv2(x))))
        x = F.silu(self.bn3(self.conv3(x)))
        x = x * self.attention(x)
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        x = torch.cat([avg_pool, max_pool], dim=1).flatten(1)
        return self.proj(x)


class Tac3DMLP(nn.Module):
    """Encode flattened Tac3D signals such as wrench or raw taxel vectors."""

    def __init__(self, input_dim: int, feature_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(1)
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Tac3D MLP input expected last dim {self.input_dim}, got {x.shape[-1]}.")
        return self.net(x)


class TactileTokenEncoder(nn.Module):
    """Project Tac3D readings to one or more policy prefix tokens."""

    def __init__(
        self,
        encoder_type: str,
        input_shape: tuple[int, int],
        input_channels: int,
        raw_shape: tuple[int, int],
        feature_dim: int,
        n_tokens: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.feature_dim = feature_dim

        if encoder_type == "tac3d_cnn":
            self.backbone = Tac3DCNN(input_shape, input_channels, feature_dim, dropout)
        elif encoder_type == "tac3d_attention":
            self.backbone = Tac3DAttentionCNN(input_shape, input_channels, feature_dim, dropout)
        elif encoder_type == "mlp":
            self.backbone = Tac3DMLP(raw_shape[0] * raw_shape[1], feature_dim, dropout)
        else:
            raise ValueError(f"Unknown tactile encoder type: {encoder_type!r}.")

        self.token_proj = nn.Linear(feature_dim, n_tokens * feature_dim) if n_tokens > 1 else None

    def forward(self, x: Tensor) -> Tensor:
        feat = self.backbone(x)
        if self.n_tokens == 1:
            return feat[:, None, :]
        feat = self.token_proj(feat)
        return feat.view(feat.shape[0], self.n_tokens, self.feature_dim)

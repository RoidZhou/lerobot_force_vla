"""Temporal encoder for force/torque windows."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Sequential:
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.GELU(),
    )


class ForceEncoder(nn.Module):
    def __init__(
        self,
        window: int = 16,
        force_dim: int = 6,
        hidden_channels: int = 128,
        bottleneck_channels: int = 256,
        embed_dim: int = 256,
        n_strided_blocks: int = 2,
    ):
        super().__init__()
        self.window = window
        self.force_dim = force_dim
        self.embed_dim = embed_dim

        self.stem = _conv_block(force_dim, hidden_channels, kernel=5, stride=1)
        blocks: List[nn.Module] = []
        cur_t = window
        cur_ch = hidden_channels
        for i in range(n_strided_blocks):
            stride = 2 if cur_t >= 4 else 1
            out_ch = bottleneck_channels if i == n_strided_blocks - 1 else hidden_channels
            blocks.append(_conv_block(cur_ch, out_ch, kernel=5, stride=stride))
            cur_ch = out_ch
            if stride > 1:
                cur_t //= stride
        self.strided = nn.Sequential(*blocks)
        self._bottleneck_t = cur_t
        self.proj = nn.Conv1d(cur_ch, embed_dim, kernel_size=3, padding=1)

    @property
    def bottleneck_t(self) -> int:
        return self._bottleneck_t

    def forward(self, force: torch.Tensor) -> torch.Tensor:
        """force: [B, T, D] -> z_e: [B, embed_dim]."""
        if force.dim() != 3:
            raise ValueError(f"Expected force [B, T, D], got {tuple(force.shape)}")
        _, time, dim = force.shape
        if time != self.window:
            raise ValueError(f"Encoder built for window={self.window}, got T={time}")
        if dim != self.force_dim:
            raise ValueError(f"Encoder built for force_dim={self.force_dim}, got D={dim}")

        x = force.transpose(1, 2).contiguous()
        x = self.stem(x)
        x = self.strided(x)
        x = self.proj(x)
        return x.mean(dim=2)


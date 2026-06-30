"""Temporal decoder for force/torque VQ-VAE latents."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _upconv_block(in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Sequential:
    pad = kernel // 2
    if stride > 1:
        layer = nn.ConvTranspose1d(
            in_ch,
            out_ch,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            output_padding=stride - 1,
        )
    else:
        layer = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=1, padding=pad)
    return nn.Sequential(
        layer,
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.GELU(),
    )


class ForceDecoder(nn.Module):
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

        cur_t = window
        strides: List[int] = []
        ch_chain = [hidden_channels]
        for i in range(n_strided_blocks):
            stride = 2 if cur_t >= 4 else 1
            strides.append(stride)
            ch_chain.append(bottleneck_channels if i == n_strided_blocks - 1 else hidden_channels)
            if stride > 1:
                cur_t //= stride
        self._bottleneck_t = cur_t

        self.from_embed = nn.Conv1d(embed_dim, bottleneck_channels, kernel_size=3, padding=1)

        blocks: List[nn.Module] = []
        in_ch = bottleneck_channels
        rev_strides = list(reversed(strides))
        rev_ch_chain = list(reversed(ch_chain))
        for i, stride in enumerate(rev_strides):
            out_ch = rev_ch_chain[i + 1]
            blocks.append(_upconv_block(in_ch, out_ch, kernel=5, stride=stride))
            in_ch = out_ch
        self.up_strided = nn.Sequential(*blocks)
        self.head = nn.Conv1d(hidden_channels, force_dim, kernel_size=5, padding=2)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """z_q: [B, embed_dim] -> recon: [B, T, D]."""
        x = z_q.unsqueeze(2).expand(-1, -1, self._bottleneck_t)
        x = self.from_embed(x)
        x = self.up_strided(x)
        x = self.head(x)
        if x.shape[-1] != self.window:
            if x.shape[-1] > self.window:
                x = x[..., : self.window]
            else:
                x = F.pad(x, (0, self.window - x.shape[-1]))
        return x.transpose(1, 2).contiguous()


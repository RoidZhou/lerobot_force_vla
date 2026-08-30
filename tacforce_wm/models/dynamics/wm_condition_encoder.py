import torch
from torch import nn
import torch.nn.functional as F


def causal_left_pad_1d(x, kernel_size, dilation=1):
    pad = (kernel_size - 1) * dilation
    return F.pad(x, (pad, 0))


class CausalConvBlock1d(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.norm = nn.GroupNorm(1, dim)
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=1,
            padding=0,
        )
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = self.norm(x)
        h = self.act(h)
        h = causal_left_pad_1d(h, self.kernel_size, self.dilation)
        h = self.conv(h)
        h = self.dropout(h)
        return x + h


class CausalDownsample1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x):
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)
        return x


class BranchEncoder1d(nn.Module):
    """
    Per-branch temporal encoder:
      [B, T4, D_in] -> [B, T, D_model]
    """
    def __init__(
        self,
        in_dim,
        model_dim,
        hidden_dim=None,
        depth=2,
        kernel_size=3,
        dropout=0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or model_dim

        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList([
            CausalConvBlock1d(
                dim=hidden_dim,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=dropout,
            )
            for i in range(depth)
        ])

        self.downsample = CausalDownsample1d(
            in_dim=hidden_dim,
            out_dim=model_dim,
            kernel_size=4,
            stride=4,
        )

        self.out_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        """
        x: [B, T4, D_in]
        return: [B, T, D_model]
        """
        x = self.input_proj(x)         # [B, T4, H]
        x = x.transpose(1, 2)          # [B, H, T4]

        for blk in self.blocks:
            x = blk(x)

        x = self.downsample(x)         # [B, D_model, T]
        x = x.transpose(1, 2)          # [B, T, D_model]
        x = self.out_norm(x)
        return x


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        force_dim=6,
        state_dim=9,
        model_dim=64,
        hidden_dim=96,
        dropout=0.0,
        use_force=True,
        use_state=False,
        use_delta_force=False,
        use_delta_state=False,
        branch_depth=2,
        kernel_size=5,
    ):
        super().__init__()

        self.force_encoder = None
        self.state_encoder = None
        self.delta_force_encoder = None
        self.delta_state_encoder = None

        if use_force:
            self.force_encoder = BranchEncoder1d(
                in_dim=force_dim,
                model_dim=model_dim,
                hidden_dim=hidden_dim,
                depth=branch_depth,
                kernel_size=kernel_size,
                dropout=dropout,
            )

        if use_state:
            self.state_encoder = BranchEncoder1d(
                in_dim=state_dim,
                model_dim=model_dim,
                hidden_dim=hidden_dim,
                depth=branch_depth,
                kernel_size=kernel_size,
                dropout=dropout,
            )

        if use_delta_force:
            self.delta_force_encoder = BranchEncoder1d(
                in_dim=force_dim,
                model_dim=model_dim,
                hidden_dim=hidden_dim,
                depth=branch_depth,
                kernel_size=kernel_size,
                dropout=dropout,
            )

        if use_delta_state:
            self.delta_state_encoder = BranchEncoder1d(
                in_dim=state_dim,
                model_dim=model_dim,
                hidden_dim=hidden_dim,
                depth=branch_depth,
                kernel_size=kernel_size,
                dropout=dropout,
            )

        num_streams = int(use_force) + int(use_state) + int(use_delta_force) + int(use_delta_state)
        if num_streams == 0:
            raise ValueError("At least one condition stream must be enabled.")

        fuse_in = num_streams * model_dim
        self.fuse = nn.Sequential(
            nn.LayerNorm(fuse_in),
            nn.Linear(fuse_in, 2 * model_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * model_dim, model_dim),
        )

        self.out_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        force_4x=None,
        state_4x=None,
        delta_force_4x=None,
        delta_state_4x=None,
    ):
        parts = []

        if self.force_encoder is not None:
            if force_4x is None:
                raise ValueError("force_4x is required but missing.")
            parts.append(self.force_encoder(force_4x.float()))

        if self.state_encoder is not None:
            if state_4x is None:
                raise ValueError("state_4x is required but missing.")
            parts.append(self.state_encoder(state_4x.float()))

        if self.delta_force_encoder is not None:
            if delta_force_4x is None:
                raise ValueError("delta_force_4x is required but missing.")
            parts.append(self.delta_force_encoder(delta_force_4x.float()))

        if self.delta_state_encoder is not None:
            if delta_state_4x is None:
                raise ValueError("delta_state_4x is required but missing.")
            parts.append(self.delta_state_encoder(delta_state_4x.float()))

        cond = torch.cat(parts, dim=-1)    # [B, T, K*D]
        cond = self.fuse(cond)             # [B, T, D]
        cond = self.out_norm(cond)
        return cond
"""Frozen PoseNet adapter and nut-local pose math used by FR-VLA.

The preprocessing and pose convention intentionally match
``ur5_rg2_infer_tactile_pose_nutbolt.py``: T_nut_fingertip is represented as
[x, y, z, roll, pitch, yaw] in metres/radians (ZYX RPY convention).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def rpy_to_rotation(rpy: Tensor) -> Tensor:
    roll, pitch, yaw = rpy.unbind(-1)
    cr, sr = roll.cos(), roll.sin()
    cp, sp = pitch.cos(), pitch.sin()
    cy, sy = yaw.cos(), yaw.sin()
    return torch.stack((
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp, cp * sr, cp * cr,
    ), dim=-1).reshape(*rpy.shape[:-1], 3, 3)


def rotation_vector(rotation: Tensor) -> Tensor:
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    skew = torch.stack((
        rotation[..., 2, 1] - rotation[..., 1, 2],
        rotation[..., 0, 2] - rotation[..., 2, 0],
        rotation[..., 1, 0] - rotation[..., 0, 1],
    ), dim=-1)
    scale = angle / (2.0 * torch.sin(angle).clamp_min(1e-7))
    return torch.where((angle < 1e-4)[..., None], 0.5 * skew, scale[..., None] * skew)


def pose_error(current_pose: Tensor, reference_pose: Tensor) -> Tensor:
    """Exact torch equivalent of NutLocalPosePID's pose error."""
    position_error = reference_pose[..., :3] - current_pose[..., :3]
    reference_rotation = rpy_to_rotation(reference_pose[..., 3:6])
    current_rotation = rpy_to_rotation(current_pose[..., 3:6])
    rotation_error = rotation_vector(reference_rotation @ current_rotation.transpose(-1, -2))
    return torch.cat((position_error, rotation_error), dim=-1)


class FrozenPoseNet(nn.Module):
    """Load and run the exact PoseNet architecture/checkpoint used by xxx."""

    def __init__(self, checkpoint_path: str, module_root: str, sensor_grid_size: int = 32,
                 sensor_shape: tuple[int, int] | None = None, allow_sensor_resize: bool = False):
        super().__init__()
        root = str(Path(module_root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from tactile_policy.model import create_model

        checkpoint = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location="cpu", weights_only=False)
        model_type = checkpoint.get("model_type", "simple_cnn")
        if model_type == "tactile_servo_control_simple_cnn":
            model_type = "simple_cnn"
        self.grid_size = int(checkpoint.get("grid_size", 32))
        self.sensor_shape = tuple(sensor_shape or (int(sensor_grid_size), int(sensor_grid_size)))
        self.allow_sensor_resize = bool(allow_sensor_resize)
        self.model = create_model(
            model_type,
            grid_size=self.grid_size,
            image_size=int(checkpoint.get("model_image_size", 128)),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        norm = checkpoint["normalization"]
        for name in ("input_mean", "input_std", "target_mean", "target_std"):
            self.register_buffer(name, torch.as_tensor(norm[name], dtype=torch.float32))
        self.target_frame = checkpoint.get("target_frame", "world")
        if self.target_frame not in {"nut", "nut_local"}:
            raise ValueError(f"PoseNet must predict nut-local pose, got target_frame={self.target_frame!r}")
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, tactile: Tensor) -> Tensor:
        # tactile: [B,2,N,3] or [B,2,H,W,3], take the current/latest frame upstream.
        if tactile.ndim == 4:
            side = int(round(tactile.shape[-2] ** 0.5))
            if side * side != tactile.shape[-2]:
                raise ValueError(f"PoseNet requires a square tactile grid, got {tactile.shape}")
            tactile = tactile.reshape(tactile.shape[0], 2, side, side, 3)
        if tactile.ndim != 5 or tactile.shape[1] != 2 or tactile.shape[-1] != 3:
            raise ValueError(f"Expected PoseNet tactile [B,2,H,W,3], got {tuple(tactile.shape)}")
        x = tactile.permute(0, 1, 4, 2, 3).flatten(1, 2)
        if x.shape[-2:] != self.sensor_shape:
            if not self.allow_sensor_resize:
                raise ValueError(
                    f"PoseNet expects sensor input {self.sensor_shape}, "
                    f"got {tuple(x.shape[-2:])}. Use a compatible PoseNet/data field or explicitly set "
                    "posenet_allow_sensor_resize=true."
                )
            x = F.interpolate(x, self.sensor_shape, mode="bilinear", align_corners=False)
        x = torch.sign(x) * torch.log1p(torch.abs(x))
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        # Keep xxx/PoseNet behavior: its regressor pads rectangular maps to a
        # square and resizes internally before invoking the tokenizer.
        output = self.model(x)
        return output * self.target_std + self.target_mean


class FrozenPoseTactileTokenizer(nn.Module):
    """Pretrained TactileTokenizer taken from the PoseNet checkpoint."""

    def __init__(self, checkpoint_path: str, module_root: str, sensor_grid_size: int = 32,
                 sensor_shape: tuple[int, int] | None = None, allow_sensor_resize: bool = False,
                 output: str = "tokens", frozen: bool = True):
        super().__init__()
        root = str(Path(module_root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from tactile_policy.model import create_model

        checkpoint = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location="cpu", weights_only=False)
        if checkpoint.get("model_type") != "tactile_tokenizer":
            raise ValueError(
                "The tactile tokenizer checkpoint must have model_type='tactile_tokenizer'; "
                f"got {checkpoint.get('model_type')!r}."
            )
        self.grid_size = int(checkpoint.get("grid_size", 32))
        self.sensor_shape = tuple(sensor_shape or (int(sensor_grid_size), int(sensor_grid_size)))
        self.allow_sensor_resize = bool(allow_sensor_resize)
        if output not in {"global", "tokens"}:
            raise ValueError("TactileTokenizer output must be 'global' or 'tokens'.")
        self.output = output
        self.frozen = bool(frozen)
        pose_model = create_model(
            "tactile_tokenizer", grid_size=self.grid_size,
            image_size=int(checkpoint.get("model_image_size", 128)),
        )
        pose_model.load_state_dict(checkpoint["model_state_dict"])
        self.tokenizer = pose_model.tokenizer
        self.embed_dim = int(self.tokenizer.embed_dim)
        norm = checkpoint["normalization"]
        self.register_buffer("input_mean", torch.as_tensor(norm["input_mean"], dtype=torch.float32))
        self.register_buffer("input_std", torch.as_tensor(norm["input_std"], dtype=torch.float32))
        self.requires_grad_(not self.frozen)
        self.eval()

    def train(self, mode: bool = True):
        super().train(mode and not self.frozen)
        self.tokenizer.train(mode and not self.frozen)
        return self

    def forward(self, tactile: Tensor) -> Tensor:
        if tactile.ndim == 4:
            side = int(round(tactile.shape[-2] ** 0.5))
            if side * side != tactile.shape[-2]:
                raise ValueError(f"TactileTokenizer requires a square tactile grid, got {tactile.shape}")
            tactile = tactile.reshape(tactile.shape[0], 2, side, side, 3)
        if tactile.ndim != 5 or tactile.shape[1] != 2 or tactile.shape[-1] != 3:
            raise ValueError(f"Expected tokenizer tactile [B,2,H,W,3], got {tuple(tactile.shape)}")
        x = tactile.permute(0, 1, 4, 2, 3).flatten(1, 2)
        if x.shape[-2:] != self.sensor_shape:
            if not self.allow_sensor_resize:
                raise ValueError(
                    f"PoseNet TactileTokenizer expects input {self.sensor_shape}, "
                    f"got {tuple(x.shape[-2:])}. Set tactile_tokenizer_allow_sensor_resize=true only "
                    "when this conversion matches tokenizer training."
                )
            x = F.interpolate(x, self.sensor_shape, mode="bilinear", align_corners=False)
        x = torch.sign(x) * torch.log1p(torch.abs(x))
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        # Reproduce TactileTokenizerPoseRegressor.forward before calling its
        # tokenizer submodule directly: pad HxW to square, then bilinear resize.
        height, width = x.shape[-2:]
        if height != width:
            size = max(height, width)
            pad_h, pad_w = size - height, size - width
            x = F.pad(
                x,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                mode="constant", value=0.0,
            )
        if x.shape[-2:] != (self.grid_size, self.grid_size):
            x = F.interpolate(x, (self.grid_size, self.grid_size), mode="bilinear", align_corners=False)
        # PoseNet tokenizer expects [B,T,H,W,6].
        tokenizer_input = x.permute(0, 2, 3, 1)[:, None]
        encoded = self.tokenizer(tokenizer_input)
        if self.output == "global":
            return encoded["global"]
        return encoded["tokens"][:, 0]

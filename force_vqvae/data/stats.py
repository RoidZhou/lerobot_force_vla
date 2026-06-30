"""Normalization stats for force/torque VQ-VAE pretraining."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ForceStats:
    force_min: np.ndarray
    force_max: np.ndarray
    force_mask: np.ndarray

    @classmethod
    def from_lerobot_dataset(cls, dataset, force_key: str = "force_torque", force_dim: int = 6) -> "ForceStats":
        force_key = resolve_force_key(dataset, force_key)
        stats = getattr(dataset.meta, "stats", {}).get(force_key)
        if stats is not None and "min" in stats and "max" in stats:
            force_min = _to_numpy(stats["min"]).reshape(-1).astype(np.float32)
            force_max = _to_numpy(stats["max"]).reshape(-1).astype(np.float32)
        else:
            force_min, force_max = cls._scan_min_max(dataset, force_key, force_dim)

        if force_min.shape[0] != force_dim or force_max.shape[0] != force_dim:
            raise ValueError(
                f"Expected `{force_key}` stats dim {force_dim}, got {force_min.shape}/{force_max.shape}."
            )
        mask = (force_max - force_min) > 1e-8
        return cls(force_min=force_min, force_max=force_max, force_mask=mask)

    @staticmethod
    def _scan_min_max(dataset, force_key: str, force_dim: int) -> tuple[np.ndarray, np.ndarray]:
        force_min = np.full(force_dim, np.inf, dtype=np.float32)
        force_max = np.full(force_dim, -np.inf, dtype=np.float32)
        for i in range(len(dataset.hf_dataset)):
            value = _to_numpy(dataset.hf_dataset[i][force_key]).reshape(-1).astype(np.float32)
            if value.shape[0] != force_dim:
                raise ValueError(f"Expected `{force_key}` dim {force_dim}, got {value.shape[0]} at frame {i}.")
            force_min = np.minimum(force_min, value)
            force_max = np.maximum(force_max, value)
        return force_min, force_max

    def normalize(self, x: np.ndarray) -> np.ndarray:
        orig_shape = x.shape
        flat = x.reshape(-1, self.force_min.shape[0]).astype(np.float32, copy=False)
        denom = (self.force_max - self.force_min) + 1e-8
        normed = np.clip(2.0 * (flat - self.force_min) / denom - 1.0, -1.0, 1.0)
        out = np.where(self.force_mask, normed, flat)
        return out.reshape(orig_shape)

    def denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        orig_shape = x_norm.shape
        flat = x_norm.reshape(-1, self.force_min.shape[0]).astype(np.float32, copy=False)
        out = (flat + 1.0) * 0.5 * (self.force_max - self.force_min) + self.force_min
        out = np.where(self.force_mask, out, flat)
        return out.reshape(orig_shape)

    def to_dict(self) -> dict:
        return {
            "force_min": self.force_min.tolist(),
            "force_max": self.force_max.tolist(),
            "force_mask": self.force_mask.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ForceStats":
        return cls(
            force_min=np.array(data["force_min"], dtype=np.float32),
            force_max=np.array(data["force_max"], dtype=np.float32),
            force_mask=np.array(data["force_mask"], dtype=bool),
        )

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ForceStats":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def resolve_force_key(dataset, force_key: str = "force_torque") -> str:
    """Resolve a user-facing force key to the actual LeRobot parquet column."""
    feature_keys = set(getattr(dataset.meta, "features", {}).keys())
    column_keys = set(getattr(dataset.hf_dataset, "column_names", []) or [])
    available = feature_keys | column_keys
    candidates = [
        force_key,
        f"observation.{force_key}",
        force_key.replace("observation.", ""),
        "force_torque",
        "observation.force_torque",
        "observation.force",
        "force",
        "effort",
    ]
    for candidate in candidates:
        if candidate in available and candidate in column_keys:
            return candidate
    for candidate in candidates:
        if candidate in column_keys:
            return candidate
    raise KeyError(
        f"Could not find force key `{force_key}` in LeRobot parquet columns. "
        f"Available columns: {sorted(column_keys)}. Meta features: {sorted(feature_keys)}."
    )

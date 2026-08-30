from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class FieldNormalizer:
    scale: torch.Tensor
    offset: torch.Tensor

    @classmethod
    def from_data_limits(
        cls,
        data: np.ndarray,
        output_min: float = -1.0,
        output_max: float = 1.0,
        eps: float = 1e-6,
    ) -> "FieldNormalizer":
        x = np.asarray(data, dtype=np.float32).reshape(-1, data.shape[-1])
        x_min = x.min(axis=0)
        x_max = x.max(axis=0)
        x_range = np.maximum(x_max - x_min, eps)
        scale = (output_max - output_min) / x_range
        offset = output_min - scale * x_min
        return cls(
            scale=torch.from_numpy(scale.astype(np.float32)),
            offset=torch.from_numpy(offset.astype(np.float32)),
        )

    def to(self, device: torch.device):
        self.scale = self.scale.to(device)
        self.offset = self.offset.to(device)
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale.to(x.device, x.dtype)
        offset = self.offset.to(x.device, x.dtype)
        return x * scale + offset

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale.to(x.device, x.dtype)
        offset = self.offset.to(x.device, x.dtype)
        return (x - offset) / scale

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "scale": self.scale.detach().cpu(),
            "offset": self.offset.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, torch.Tensor]) -> "FieldNormalizer":
        return cls(scale=state["scale"], offset=state["offset"])


class MultiFieldNormalizer:
    def __init__(self):
        self.fields: Dict[str, FieldNormalizer] = {}

    def __contains__(self, key: str) -> bool:
        return key in self.fields

    def __getitem__(self, key: str) -> FieldNormalizer:
        return self.fields[key]

    def __setitem__(self, key: str, value: FieldNormalizer):
        self.fields[key] = value

    def to(self, device: torch.device):
        for field in self.fields.values():
            field.to(device)
        return self

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {k: v.state_dict() for k, v in self.fields.items()}

    def load_state_dict(self, state: Dict[str, Dict[str, torch.Tensor]]):
        self.fields = {
            k: FieldNormalizer.from_state_dict(v)
            for k, v in state.items()
            if isinstance(v, dict) and "scale" in v and "offset" in v
        }

    def normalize_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in obs.items():
            if k in self.fields and torch.is_tensor(v):
                out[k] = self.fields[k].normalize(v)
            else:
                out[k] = v
        return out

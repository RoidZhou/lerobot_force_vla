"""Frozen TacForce world model wrapper for policy training and inference."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import yaml

try:  # Imported as lerobot.tacforce_wm.models.dynamics.frozen.
    from .world_model import TacForceWorldModel
    from ...utils.normalizer import MultiFieldNormalizer
except ImportError:  # Backward compatibility with train.py run from tacforce_wm/.
    from models.dynamics.world_model import TacForceWorldModel
    from utils.normalizer import MultiFieldNormalizer


def _to_ns(x: Any) -> Any:
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()})
    return [_to_ns(v) for v in x] if isinstance(x, list) else x


def _read_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {path!r}")
    return raw


def _resolve_config_path(path: str) -> str:
    path = os.path.expanduser(str(path).strip())
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for name in ("resolved_config.yaml", "config.yaml"):
            cand = os.path.join(path, name)
            if os.path.isfile(cand):
                return cand
        raise FileNotFoundError(f"No config yaml found under directory {path!r}")
    raise FileNotFoundError(f"config_path not found: {path!r}")


def _wm_args_from_run_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Extract TacForceWorldModel kwargs from a dynamics training resolved config."""
    model = raw.get("model")
    if not isinstance(model, dict):
        raise ValueError("Dynamics run config must contain a 'model' section.")
    dynamics = model.get("dynamics")
    if not isinstance(dynamics, dict):
        raise ValueError("Dynamics run config must contain model.dynamics.")

    args = dict(dynamics)
    data = raw.get("data")
    if isinstance(data, dict):
        args.setdefault("window_size", data.get("window_size"))
    return args


def _ckpt_path(backend: Mapping[str, Any]) -> str:
    path = backend.get("ckpt_path")
    if isinstance(path, str) and path.strip():
        return path.strip()
    raise KeyError(
        "Set model.backend.ckpt_path to a dynamics checkpoint "
        "(e.g. outputs/<run>/checkpoints/best_model.pt)."
    )


def _load_checkpoint(path: str, device: torch.device) -> tuple[dict, dict]:
    raw = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise KeyError(f"{path} must be a dict with 'model' and 'normalizer' keys.")
    norm = raw.get("normalizer")
    if not isinstance(norm, dict):
        raise KeyError(f"{path} is missing 'normalizer'. Retrain dynamics or use a newer checkpoint.")
    return raw["model"], norm


class BaseDynamics(ABC, nn.Module):
    @property
    @abstractmethod
    def latent_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        *,
        compute_predict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class FrozenTacForceDynamics(BaseDynamics):
    """Loads a trained TacForceWorldModel, keeps it frozen, and exposes tactile latents."""

    def __init__(
        self,
        backend: Mapping[str, Any],
        *,
        device: str = "cuda",
        bundle: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        backend = dict(backend)
        bundle = dict(bundle or {})

        wm_cfg = bundle.get("wm_config")
        if isinstance(wm_cfg, dict) and wm_cfg:
            wm_args = dict(wm_cfg)
        else:
            config_path = backend.get("config_path")
            if not config_path:
                raise KeyError(
                    "model.backend.config_path is required unless dynamics_bundle embeds wm_config."
                )
            raw_cfg = _read_yaml(_resolve_config_path(str(config_path)))
            wm_args = _wm_args_from_run_config(raw_cfg)

        self.normalize_keys = list(
            backend.get("normalize_keys")
            or wm_args.get("normalize_keys")
            or ["tactile", "force_4x", "state_4x"]
        )

        self.wm = TacForceWorldModel(_to_ns(wm_args)).to(self.device)

        wm_sd = bundle.get("wm_state_dict")
        norm_sd = bundle.get("normalizer_state_dict")
        if isinstance(wm_sd, dict) and wm_sd:
            self.wm.load_state_dict(
                {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in wm_sd.items()},
                strict=True,
            )
        else:
            ckpt = _ckpt_path(backend)
            model_sd, ckpt_norm = _load_checkpoint(ckpt, self.device)
            self.wm.load_state_dict(model_sd, strict=True)
            if not (isinstance(norm_sd, dict) and norm_sd):
                norm_sd = ckpt_norm

        self.wm.eval()
        for p in self.wm.parameters():
            p.requires_grad = False

        self._latent_dim = int(self.wm.tactile_embed_dim)

        if not isinstance(norm_sd, dict):
            raise KeyError(
                "Normalizer state is required (from dynamics_bundle or the dynamics checkpoint)."
            )
        self.normalizer = MultiFieldNormalizer()
        self.normalizer.load_state_dict(norm_sd)
        self.normalizer.to(self.device)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def train(self, mode: bool = True):
        super().train(mode)
        # The pretrained tokenizer and world model must stay deterministic and
        # frozen even when the surrounding policy enters training mode.
        self.wm.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        *,
        compute_predict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        batch = {k: obs[k] for k in ("tactile", "force_4x")}
        for k in ("state_4x", "delta_force_4x", "delta_state_4x"):
            if k in obs:
                batch[k] = obs[k]
        device = next(self.wm.parameters()).device
        batch = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        for k in self.normalize_keys:
            if k in batch:
                batch[k] = self.normalizer[k].normalize(batch[k])

        enc = self.wm.encode(batch)
        z, co = enc["z"].float(), enc["cond"].float()
        if compute_predict:
            fut = self.wm.predict(z, co).float()
        else:
            fut = torch.zeros_like(z)
        return {"tactile_latent_curr": z, "tactile_latent_future": fut}


def build_dynamics(
    backend: Mapping[str, Any] | None,
    device: str,
    bundle: Optional[Mapping[str, Any]] = None,
) -> Optional[FrozenTacForceDynamics]:
    if backend is None:
        return None
    backend = dict(backend)
    bundle = dict(bundle or {})
    has_wm_config = isinstance(bundle.get("wm_config"), dict) and bool(bundle["wm_config"])
    if not backend.get("config_path") and not has_wm_config:
        raise KeyError("model.backend.config_path is required.")
    return FrozenTacForceDynamics(backend, device=device, bundle=bundle)

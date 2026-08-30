from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from models.policy.flow_policy import FlowMatchingPolicy
from trainers.policy_trainer import build_policy
from utils.train_utils import build_canonical_config


def _strip_state_dict_prefix(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    plen = len(prefix)
    return {k[plen:] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def load_policy_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Tuple[FlowMatchingPolicy, dict]:
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        raise KeyError("Checkpoint missing 'config' dict.")
    cfg = build_canonical_config("policy", copy.deepcopy(cfg))

    sd_raw = ckpt.get("policy_state_dict")
    if not isinstance(sd_raw, dict):
        raise KeyError("Checkpoint missing policy_state_dict")
    sd = _strip_state_dict_prefix(sd_raw, "module.")

    dynamics_bundle = ckpt.get("dynamics_bundle")
    policy = build_policy(cfg, device, train_dataset=None, dynamics_bundle=dynamics_bundle)

    dyn_sd = {k: v for k, v in policy.state_dict().items() if k.startswith("dynamics.")}
    incompat = policy.load_state_dict(sd, strict=False)
    missing = [k for k in incompat.missing_keys if not k.startswith("dynamics.")]
    unexpected = [k for k in incompat.unexpected_keys if not k.startswith("dynamics.")]
    if missing or unexpected:
        raise RuntimeError(
            f"Policy checkpoint incompatible: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    policy.load_state_dict(dyn_sd, strict=False)

    norm_sd = ckpt.get("normalizer_state_dict")
    if isinstance(norm_sd, dict) and norm_sd:
        norm_sd = _strip_state_dict_prefix(norm_sd, "module.")
        policy.normalizer.load_state_dict(norm_sd)
        policy.normalizer.to(device)

    policy.eval()
    return policy, cfg

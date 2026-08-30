from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.policy_zarr_dataset import ZarrDataset  # noqa: E402
from tools.action_util import relative_actions_to_absolute_tensor  # noqa: E402
from utils.checkpoint_util import load_policy_from_checkpoint  # noqa: E402
from utils.tensor import move_to_device  # noqa: E402
from utils.train_utils import cfg_get  # noqa: E402


def _dataset_kwargs_from_cfg(cfg: dict, data_root_override: Optional[str]) -> dict[str, Any]:
    data_cfg = dict(cfg["data"])
    if data_root_override:
        data_cfg["root_dir"] = str(data_root_override)
        data_cfg["latent_cache_root_dir"] = str(data_root_override)
    if data_cfg.get("n_image_steps") is None:
        data_cfg["n_image_steps"] = cfg_get(cfg, "model.policy.curr_steps", 1)
    if data_cfg.get("action_representation") is None:
        data_cfg["action_representation"] = cfg_get(cfg, "model.policy.action_representation", "absolute")
    return data_cfg


def _to_absolute_actions(
    action_seq: torch.Tensor,
    obs_batch: dict[str, torch.Tensor],
    action_representation: str,
) -> torch.Tensor:
    if str(action_representation).lower() == "absolute":
        return action_seq
    if "state" not in obs_batch:
        raise KeyError("chunk_relative conversion requires obs['state'].")

    action_dim = action_seq.shape[-1]
    state = obs_batch["state"]
    if state.shape[-1] < action_dim:
        raise ValueError(f"state_dim={state.shape[-1]} smaller than action_dim={action_dim}")

    base = state[:, -1, :action_dim].unsqueeze(1).expand(-1, action_seq.shape[1], -1)
    return relative_actions_to_absolute_tensor(action_seq, base)


def _summarize_metrics(metrics: dict[str, float], prefix: str) -> None:
    parts = [f"{k}={v:.6f}" for k, v in metrics.items()]
    print(f"[{prefix}] " + ", ".join(parts))


@torch.no_grad()
def evaluate_open_loop(
    policy: torch.nn.Module,
    ds: ZarrDataset,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int,
    num_inference_steps: int,
    solver: str,
) -> dict[str, float]:
    policy.eval()
    action_representation = str(getattr(ds, "action_representation", "absolute")).lower()
    policy_action_mse_sum = 0.0
    policy_action_l1_sum = 0.0
    sample_count = 0
    policy_xyz_errors: list[np.ndarray] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        batch_d = move_to_device(batch, device)
        out = policy.predict_action(
            batch_d["obs"],
            num_inference_steps=num_inference_steps,
            solver=solver,
        )

        pred_model = out.get("action_model", out["action"])
        gt_model = batch_d["action"]
        t_model = min(pred_model.shape[1], gt_model.shape[1])
        d_model = min(pred_model.shape[2], gt_model.shape[2])
        if t_model > 0 and d_model > 0:
            pred_model_slice = pred_model[:, :t_model, :d_model]
            gt_model_slice = gt_model[:, :t_model, :d_model]
            bs = gt_model_slice.shape[0]
            policy_action_mse_sum += float(F.mse_loss(pred_model_slice, gt_model_slice, reduction="mean").item()) * bs
            policy_action_l1_sum += float(torch.mean(torch.abs(pred_model_slice - gt_model_slice)).item()) * bs
            sample_count += bs

        pred_abs = out["action"]
        gt_abs = _to_absolute_actions(batch_d["action"], batch_d["obs"], action_representation)
        t = min(pred_abs.shape[1], gt_abs.shape[1])
        d = min(pred_abs.shape[2], gt_abs.shape[2], 3)
        if t <= 0 or d <= 0:
            continue

        pred_xyz = pred_abs[:, :t, :d]
        gt_xyz = gt_abs[:, :t, :d]
        policy_err = torch.mean(torch.abs(pred_xyz - gt_xyz), dim=(1, 2))
        policy_xyz_errors.append(policy_err.detach().cpu().numpy())

    metrics = {
        "action_mse": policy_action_mse_sum / max(sample_count, 1),
        "action_l1": policy_action_l1_sum / max(sample_count, 1),
        "policy_chunk_mae_xyz": 0.0,
    }
    if policy_xyz_errors:
        policy_err_all = np.concatenate(policy_xyz_errors, axis=0).astype(np.float64, copy=False)
        metrics["policy_chunk_mae_xyz"] = float(np.mean(policy_err_all))
    return metrics


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Open-loop evaluation for a trained flow-matching policy.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Policy checkpoint path")
    parser.add_argument("--data-root", type=str, default=None, help="Optional dataset root override")
    parser.add_argument("--split", type=str, default="both", choices=["train", "val", "both"])
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "heun"])
    parser.add_argument("--batch-size", type=int, default=None, help="Optional loader batch size override")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = str(args.checkpoint).strip()
    policy, cfg = load_policy_from_checkpoint(ckpt_path, device)
    data_kwargs = _dataset_kwargs_from_cfg(cfg, args.data_root)

    batch_size = int(args.batch_size or cfg_get(cfg, "train.val_batch_size", cfg_get(cfg, "train.batch_size", 32)))
    num_workers = int(cfg_get(cfg, "train.num_workers", 0))
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(cfg_get(cfg, "train.persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(cfg_get(cfg, "train.prefetch_factor", 4))

    max_batches = max(1, int(args.max_batches))
    common_kwargs = {
        "max_batches": max_batches,
        "num_inference_steps": int(args.num_inference_steps),
        "solver": str(args.solver),
    }

    print(f"[info] checkpoint: {ckpt_path}")
    print(f"[info] device: {device}")

    if args.split in ("train", "both"):
        train_ds = ZarrDataset(split="train", **data_kwargs)
        train_loader = DataLoader(train_ds, **loader_kwargs)
        print(f"[info] train_windows={len(train_ds)}")
        metrics = evaluate_open_loop(policy, train_ds, train_loader, device, **common_kwargs)
        _summarize_metrics(metrics, "open_loop/train")

    if args.split in ("val", "both"):
        val_ds = ZarrDataset(split="val", **data_kwargs)
        val_loader = DataLoader(val_ds, **loader_kwargs)
        print(f"[info] val_windows={len(val_ds)}")
        metrics = evaluate_open_loop(policy, val_ds, val_loader, device, **common_kwargs)
        _summarize_metrics(metrics, "open_loop/val")


if __name__ == "__main__":
    main()

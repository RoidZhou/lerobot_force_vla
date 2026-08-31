from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.policy_lerobot_dataset import LeRobotPolicyDataset  # noqa: E402
from models.dynamics import build_dynamics  # noqa: E402
from models.policy.condition_encoder import DinoV2SmallEncoder  # noqa: E402
from utils.train_utils import build_canonical_config  # noqa: E402


def build_dataset(cfg: dict, split: str) -> LeRobotPolicyDataset:
    data_cfg = dict(cfg["data"])
    data_cfg["latent_cache_root_dir"] = None
    data_cfg["preload_to_ram"] = False
    data_cfg.pop("expected_dynamics_ckpt_path", None)
    data_cfg.pop("expected_dino_checkpoint_path", None)
    if data_cfg.get("n_image_steps") is None:
        data_cfg["n_image_steps"] = cfg["model"]["policy"].get("curr_steps", 1)
    return LeRobotPolicyDataset(split=split, **data_cfg)


def output_path(cfg: dict, split: str, override: str | None) -> Path:
    if override:
        return Path(override.format(split=split)).expanduser().resolve()
    root = cfg["data"].get("latent_cache_root_dir")
    if not root:
        raise ValueError("Set data.latent_cache_root_dir or pass --output with a {split} placeholder.")
    return Path(root).expanduser().resolve() / split / "policy_latent_cache.zarr"


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Cache already exists: {path}. Pass --overwrite to rebuild it.")
        if path.name != "policy_latent_cache.zarr":
            raise ValueError(f"Refusing to overwrite unexpected cache path: {path}")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def dynamics_batch(dataset: LeRobotPolicyDataset, indices: list[int]) -> dict[str, torch.Tensor]:
    rows = [dataset.get_dynamics_input(index) for index in indices]
    return {
        key: torch.from_numpy(np.stack([row[key] for row in rows], axis=0))
        for key in ("tactile", "force_4x", "state_4x")
    }


def image_batch(dataset: LeRobotPolicyDataset, indices: list[int]) -> torch.Tensor:
    # [B,T,V,C,H,W]
    return torch.stack([dataset.get_image_input(index) for index in indices], dim=0)


def create_cache_arrays(data_group, count: int, curr, future, image, chunk_size: int):
    chunk_size = max(1, min(int(chunk_size), count))
    curr_array = _create_array(
        data_group,
        "tactile_latent_curr",
        shape=(count,) + curr.shape[1:],
        chunks=(chunk_size,) + curr.shape[1:],
        dtype="f4",
    )
    future_array = _create_array(
        data_group,
        "tactile_latent_future",
        shape=(count,) + future.shape[1:],
        chunks=(chunk_size,) + future.shape[1:],
        dtype="f4",
    )
    image_array = _create_array(
        data_group,
        "image_backbone_feat",
        shape=(count,) + image.shape[1:],
        chunks=(chunk_size,) + image.shape[1:],
        dtype="f4",
    )
    return curr_array, future_array, image_array


def _create_array(group, name: str, *, data=None, shape=None, chunks=None, dtype=None):
    """Write with either Zarr 2 or Zarr 3 without changing cache layout."""
    kwargs = {}
    if data is not None:
        kwargs["data"] = data
        kwargs["shape"] = np.asarray(data).shape
    elif shape is not None:
        kwargs["shape"] = shape
    if chunks is not None:
        kwargs["chunks"] = chunks
    if dtype is not None:
        kwargs["dtype"] = dtype
    if hasattr(group, "create_array"):
        if data is not None:
            kwargs.pop("shape", None)
        return group.create_array(name, **kwargs)
    return group.create_dataset(name, **kwargs)


def precompute_split(cfg: dict, split: str, args) -> Path:
    dataset = build_dataset(cfg, split)
    path = output_path(cfg, split, args.output)
    prepare_output(path, args.overwrite)

    device = torch.device(args.device)
    dynamics = build_dynamics(cfg["model"]["backend"], device=str(device))
    dynamics.eval()
    policy_cfg = cfg["model"]["policy"]
    if not bool(policy_cfg.get("freeze_image_encoder", True)):
        raise ValueError("Image feature caching requires model.policy.freeze_image_encoder=true.")
    image_encoder = DinoV2SmallEncoder(
        out_dim=policy_cfg.get("cond_dim", 256),
        pretrained=policy_cfg.get("image_pretrained", True),
        freeze=True,
        model_name=policy_cfg.get("dino_model_name", "vit_small_patch14_dinov2.lvd142m"),
        checkpoint_path=policy_cfg.get("dino_checkpoint_path"),
    ).to(device).eval()

    root = zarr.open_group(str(path), mode="w")
    dynamics_ckpt = cfg["model"]["backend"].get("ckpt_path")
    dino_ckpt = policy_cfg.get("dino_checkpoint_path")
    root.attrs.update({
        "cache_version": 2,
        "split": split,
        "source_root": dataset.root_dir,
        "window_size": dataset.window_size,
        "action_window_size": dataset.action_window_size,
        "stride": dataset.stride,
        "n_image_steps": dataset.n_image_steps,
        "tactile_left_key": dataset.tactile_left_key,
        "tactile_right_key": dataset.tactile_right_key,
        "image_keys": list(dataset.image_keys),
        "dynamics_checkpoint_sha256": file_sha256(dynamics_ckpt),
        "dino_checkpoint_sha256": file_sha256(dino_ckpt),
    })
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")
    episode_indices, anchors = dataset._window_metadata()
    _create_array(meta_group, "window_episode_indices", data=episode_indices)
    _create_array(meta_group, "window_anchor_times", data=anchors)

    arrays = None
    batch_size = max(1, int(args.batch_size))
    for start in tqdm(range(0, len(dataset), batch_size), desc=f"precompute:{split}", unit="batch"):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        dyn_input = dynamics_batch(dataset, indices)
        images = image_batch(dataset, indices).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(args.amp and device.type == "cuda"),
        ):
            dyn_output = dynamics(dyn_input)
            bsz, timesteps, views = images.shape[:3]
            image_feature = image_encoder.extract_backbone_feat(
                images.reshape(bsz * timesteps * views, *images.shape[3:])
            ).reshape(bsz, timesteps, views, -1)

        curr = dyn_output["tactile_latent_curr"].float().cpu().numpy()
        future = dyn_output["tactile_latent_future"].float().cpu().numpy()
        image = image_feature.float().cpu().numpy()
        if arrays is None:
            arrays = create_cache_arrays(data_group, len(dataset), curr, future, image, args.chunk_size)
            root.attrs.update({
                "latent_dim": int(curr.shape[-1]),
                "latent_steps": int(curr.shape[1]),
                "image_backbone_dim": int(image.shape[-1]),
            })
        end = start + len(indices)
        arrays[0][start:end] = curr
        arrays[1][start:end] = future
        arrays[2][start:end] = image

    print(f"Saved {len(dataset)} {split} windows to {path}")
    return path


def file_sha256(path: str | None) -> str:
    if not path:
        raise ValueError("A local checkpoint path is required for reproducible latent caching.")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute LeRobot wrist DINOv2 features and TacForce tactile latents."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default=None, help="Optional path supporting the {split} placeholder.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = build_canonical_config("policy", args.config)
    splits = ("train", "val") if args.split == "all" else (args.split,)
    for split in splits:
        precompute_split(cfg, split, args)


if __name__ == "__main__":
    main()

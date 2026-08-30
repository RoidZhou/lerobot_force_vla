from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import zarr
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from datasets.policy_zarr_dataset import ZarrDataset  # noqa: E402
from models.policy.condition_encoder import DinoV2SmallEncoder  # noqa: E402
from models.dynamics import build_dynamics  # noqa: E402
from utils.train_utils import build_canonical_config  # noqa: E402


def build_dataset(cfg: dict, split: str) -> ZarrDataset:
    data_cfg = dict(cfg["data"])
    if data_cfg.get("n_image_steps") is None:
        data_cfg["n_image_steps"] = cfg["model"]["policy"].get("curr_steps", 1)
    if data_cfg.get("action_representation") is None:
        data_cfg["action_representation"] = cfg["model"]["policy"].get("action_representation", "absolute")
    # Precompute should always read raw streams, not latent cache.
    data_cfg["latent_cache_root_dir"] = None
    return ZarrDataset(split=split, **data_cfg)


def resolve_output_path(dataset: ZarrDataset, cfg: dict, split: str, output_path: str | None) -> str:
    if output_path:
        return output_path.format(split=split)

    data_cfg = cfg["data"]
    if data_cfg.get("latent_cache_root_dir"):
        return os.path.join(str(data_cfg["latent_cache_root_dir"]), split, "policy_latent_cache.zarr")

    return os.path.join(os.path.dirname(dataset.zarr_path), "policy_latent_cache.zarr")


def build_obs_batch(dataset: ZarrDataset, indices: list[int]) -> dict[str, torch.Tensor]:
    tactile_batch = []
    force_batch = []
    state_batch = []

    for idx in indices:
        obs = dataset.get_dynamics_input(idx)
        tactile_batch.append(obs["tactile"])
        force_batch.append(obs["force_4x"])
        state_batch.append(obs["state_4x"])

    return {
        "tactile": torch.from_numpy(np.stack(tactile_batch, axis=0)),
        "force_4x": torch.from_numpy(np.stack(force_batch, axis=0)),
        "state_4x": torch.from_numpy(np.stack(state_batch, axis=0)),
    }


def build_image_batch(dataset: ZarrDataset, indices: list[int]) -> torch.Tensor:
    image_batch = []

    for idx in indices:
        anchor_t, _, _ = dataset.windows[idx]
        views = []
        for key in dataset.image_keys:
            image = dataset._read_array(key, int(anchor_t))
            views.append(dataset._process_image(image).numpy())
        image_batch.append(np.stack(views, axis=0))

    return torch.from_numpy(np.stack(image_batch, axis=0))


def write_window_metadata(meta_group, dataset: ZarrDataset):
    windows = np.asarray(dataset.windows, dtype=np.int64)
    meta_group.create_array("window_anchor_times", data=windows[:, 0])
    meta_group.create_array("window_episode_ends", data=windows[:, 1])
    meta_group.create_array("window_episode_indices", data=windows[:, 2])


def main():
    parser = argparse.ArgumentParser(description="Precompute policy tactile latents and frozen image backbone features.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config" / "config.yaml"),
        help="Path to config yaml",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None, help="Output zarr path. Supports {split}.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = build_canonical_config("policy", args.config)
    dataset = build_dataset(cfg, args.split)
    output_path = resolve_output_path(dataset, cfg, args.split, args.output)

    if os.path.isdir(output_path):
        if not args.overwrite:
            raise FileExistsError(f"Output zarr already exists: {output_path}. Use --overwrite to rebuild.")
        shutil.rmtree(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device = torch.device(args.device)
    dynamics = build_dynamics(cfg["model"]["backend"], device=str(device))
    dynamics.eval()
    policy_cfg = cfg["model"]["policy"]
    if not bool(policy_cfg.get("freeze_image_encoder", True)):
        raise ValueError("Offline image feature caching requires policy.freeze_image_encoder=True.")
    image_encoder = DinoV2SmallEncoder(
        out_dim=policy_cfg.get("cond_dim", 256),
        pretrained=policy_cfg.get("image_pretrained", True),
        freeze=True,
        model_name=policy_cfg.get("dino_model_name", "vit_small_patch14_dinov2.lvd142m"),
    ).to(device)
    image_encoder.eval()

    out_root = zarr.open_group(output_path, mode="w")
    out_root.attrs["cache_version"] = 1
    out_root.attrs["split"] = args.split
    out_root.attrs["source_zarr_path"] = dataset.zarr_path
    out_root.attrs["window_size"] = int(dataset.window_size)
    out_root.attrs["action_window_size"] = int(dataset.action_window_size)
    out_root.attrs["stride"] = int(dataset.stride)
    out_root.attrs["image_selection"] = "anchor"

    data_group = out_root.create_group("data")
    meta_group = out_root.create_group("meta")
    write_window_metadata(meta_group, dataset)

    curr_arr = None
    fut_arr = None
    img_arr = None
    chunk_bsz = max(1, min(args.batch_size, 1024))

    for start_idx in tqdm(range(0, len(dataset), args.batch_size), desc=f"cache:{args.split}", unit="window"):
        batch_indices = list(range(start_idx, min(start_idx + args.batch_size, len(dataset))))
        obs_batch = build_obs_batch(dataset, batch_indices)
        image_batch = build_image_batch(dataset, batch_indices).to(device, non_blocking=True)

        with torch.inference_mode():
            out = dynamics(obs_batch)
            bsz, num_views = image_batch.shape[:2]
            image_feat = image_encoder.extract_backbone_feat(
                image_batch.reshape(bsz * num_views, *image_batch.shape[2:])
            ).reshape(bsz, num_views, -1)

        curr = out["tactile_latent_curr"].detach().cpu().numpy().astype(np.float32, copy=False)
        fut = out["tactile_latent_future"].detach().cpu().numpy().astype(np.float32, copy=False)
        img = image_feat.detach().cpu().numpy().astype(np.float32, copy=False)

        if curr_arr is None:
            curr_arr = data_group.create_array(
                "tactile_latent_curr",
                shape=(len(dataset),) + curr.shape[1:],
                chunks=(chunk_bsz,) + curr.shape[1:],
                dtype="f4",
            )
            fut_arr = data_group.create_array(
                "tactile_latent_future",
                shape=(len(dataset),) + fut.shape[1:],
                chunks=(chunk_bsz,) + fut.shape[1:],
                dtype="f4",
            )
            img_arr = data_group.create_array(
                "image_backbone_feat",
                shape=(len(dataset),) + img.shape[1:],
                chunks=(chunk_bsz,) + img.shape[1:],
                dtype="f4",
            )
            out_root.attrs["latent_dim"] = int(curr.shape[-1])
            out_root.attrs["pred_steps"] = int(curr.shape[1])
            out_root.attrs["image_backbone_dim"] = int(img.shape[-1])

        curr_arr[start_idx:start_idx + len(batch_indices)] = curr
        fut_arr[start_idx:start_idx + len(batch_indices)] = fut
        img_arr[start_idx:start_idx + len(batch_indices)] = img

    print(f"Saved latent cache to: {output_path}")


if __name__ == "__main__":
    main()

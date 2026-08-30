from __future__ import annotations

import glob
import io
import os
import re
from functools import lru_cache
from typing import Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from datasets.tactile_datasets import _pack_tactile
from utils.normalizer import FieldNormalizer, MultiFieldNormalizer


_EPISODE_RE = re.compile(r"episode_(\d+)\.parquet$")


def _subsample(arr: np.ndarray, max_rows: int | None) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1, arr.shape[-1])
    if not max_rows or max_rows <= 0 or len(flat) <= max_rows:
        return flat
    ids = np.linspace(0, len(flat) - 1, int(max_rows), dtype=np.int64)
    return flat[ids]


class LeRobotPolicyDataset(Dataset):
    """Policy windows from LeRobot v2.1 parquet episodes."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        window_size: int = 16,
        stride: int = 1,
        action_dim: int = 7,
        image_size: int = 224,
        n_image_steps: int | None = 1,
        action_window_size: int | None = 32,
        image_as_uint8: bool = True,
        preload_to_ram: bool = False,
        latent_cache_root_dir: str | None = None,
        image_keys: Sequence[str] = ("observation.wrist_image",),
        tactile_left_key: str = "observation.tactile_left.displacement",
        tactile_right_key: str = "observation.tactile_right.displacement",
        force_key: str = "observation.force_torque",
        state_key: str = "observation.state",
        force_4x_key: str | None = None,
        state_4x_key: str | None = None,
        action_key: str = "action",
        action_representation: str = "absolute",
        val_fraction: float = 0.2,
        split_seed: int = 42,
        force_upsample: int = 4,
        format: str | None = None,
    ):
        del preload_to_ram, latent_cache_root_dir, force_4x_key, state_4x_key, format
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.split = split
        self.window_size = max(1, int(window_size))
        self.stride = max(1, int(stride))
        self.action_dim = int(action_dim)
        self.image_size = int(image_size)
        self.n_image_steps = max(1, int(n_image_steps or 1))
        self.action_window_size = max(1, int(action_window_size or window_size))
        self.image_as_uint8 = bool(image_as_uint8)
        self.image_keys = list(image_keys)
        self.tactile_left_key = tactile_left_key
        self.tactile_right_key = tactile_right_key
        self.force_key = force_key
        self.state_key = state_key
        self.action_key = action_key
        self.action_representation = str(action_representation).lower()
        self.force_upsample = int(force_upsample)
        if self.action_representation != "absolute":
            raise ValueError("This 6D-state/7D-action dataset requires action_representation='absolute'.")
        if self.force_upsample != 4:
            raise ValueError("The frozen TacForce condition encoder requires force_upsample=4.")
        if self.image_keys != ["observation.wrist_image"]:
            raise ValueError("This configuration is restricted to observation.wrist_image.")

        paths = sorted(glob.glob(os.path.join(self.root_dir, "data", "chunk-*", "episode_*.parquet")))
        if not paths:
            raise FileNotFoundError(f"No episode parquet files found under {self.root_dir}/data")
        all_episodes = []
        for path in paths:
            match = _EPISODE_RE.search(path)
            if match:
                all_episodes.append((int(match.group(1)), path))
        order = np.random.default_rng(int(split_seed)).permutation(len(all_episodes))
        val_count = max(1, int(round(len(all_episodes) * float(val_fraction))))
        val_ids = set(order[-val_count:].tolist())
        want_val = split in {"val", "test"}
        self.episodes = [ep for i, ep in enumerate(all_episodes) if (i in val_ids) == want_val]

        self.windows: list[tuple[int, int]] = []
        for episode_id, (_, path) in enumerate(self.episodes):
            length = pq.ParquetFile(path).metadata.num_rows
            first_anchor = max(self.window_size, self.n_image_steps) - 1
            last_anchor = length - self.action_window_size
            for anchor in range(first_anchor, last_anchor + 1, self.stride):
                self.windows.append((episode_id, anchor))
        if not self.windows:
            raise ValueError("No valid policy windows were found.")

        self.cached_image_backbone_feat = None
        self.cached_tactile_latent_curr = None
        self.cached_tactile_latent_future = None
        self.cached_latent_dim = None

    @property
    def _columns(self) -> list[str]:
        return [*self.image_keys, self.tactile_left_key, self.tactile_right_key,
                self.force_key, self.state_key, self.action_key]

    @lru_cache(maxsize=8)
    def _load_row_group(self, episode_id: int, row_group: int) -> dict:
        parquet = pq.ParquetFile(self.episodes[episode_id][1])
        table = parquet.read_row_group(row_group, columns=self._columns)
        result = {}
        for key in self._columns:
            values = table[key].to_pylist()
            result[key] = values if key in self.image_keys else np.asarray(values, dtype=np.float32)
        return result

    def _read_slice(self, episode_id: int, key: str, start: int, end: int):
        parquet = pq.ParquetFile(self.episodes[episode_id][1])
        offsets = []
        total = 0
        for group in range(parquet.num_row_groups):
            offsets.append(total)
            total += parquet.metadata.row_group(group).num_rows
        pieces = []
        for group, group_start in enumerate(offsets):
            group_end = group_start + parquet.metadata.row_group(group).num_rows
            if group_end <= start or group_start >= end:
                continue
            values = self._load_row_group(episode_id, group)[key]
            local_start = max(start, group_start) - group_start
            local_end = min(end, group_end) - group_start
            pieces.extend(values[local_start:local_end] if key in self.image_keys
                          else [values[local_start:local_end]])
        if key in self.image_keys:
            return pieces
        return np.concatenate(pieces, axis=0)

    def _process_images(self, frames: list[dict]) -> torch.Tensor:
        tensors = []
        for frame in frames:
            if not isinstance(frame, dict) or not frame.get("bytes"):
                raise ValueError("LeRobot image entry is missing encoded image bytes.")
            with Image.open(io.BytesIO(frame["bytes"])) as image:
                arr = np.asarray(image.convert("RGB")).copy()
            x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            if x.shape[-2:] != (self.image_size, self.image_size):
                x = F.interpolate(x[None].float(), (self.image_size, self.image_size),
                                  mode="bilinear", align_corners=False)[0]
                x = x.round().clamp(0, 255).to(torch.uint8)
            if not self.image_as_uint8:
                x = x.float().div(255).mul(2).sub(1)
            tensors.append(x)
        return torch.stack(tensors)

    def __len__(self) -> int:
        return len(self.windows)

    def get_dynamics_input(self, idx: int) -> dict[str, np.ndarray]:
        episode_id, anchor = self.windows[idx]
        start, end = anchor - self.window_size + 1, anchor + 1
        tactile = _pack_tactile(self._read_slice(episode_id, self.tactile_left_key, start, end),
                                self._read_slice(episode_id, self.tactile_right_key, start, end))
        return {
            "tactile": tactile,
            "force_4x": np.repeat(self._read_slice(episode_id, self.force_key, start, end), 4, axis=0),
            "state_4x": np.repeat(self._read_slice(episode_id, self.state_key, start, end), 4, axis=0),
        }

    def __getitem__(self, idx: int) -> dict:
        episode_id, anchor = self.windows[idx]
        obs_start, obs_end = anchor - self.window_size + 1, anchor + 1
        image_start = anchor - self.n_image_steps + 1
        dynamics = self.get_dynamics_input(idx)

        views = [self._process_images(self._read_slice(episode_id, key, image_start, anchor + 1))
                 for key in self.image_keys]
        # [T,V,C,H,W], where V=1 and is strictly the wrist camera.
        image = torch.stack(views, dim=1)
        obs = {
            "image": image,
            "force": torch.from_numpy(self._read_slice(episode_id, self.force_key, obs_start, obs_end)),
            "state": torch.from_numpy(self._read_slice(episode_id, self.state_key, obs_start, obs_end)),
            "tactile": torch.from_numpy(dynamics["tactile"]),
            "force_4x": torch.from_numpy(dynamics["force_4x"]),
            "state_4x": torch.from_numpy(dynamics["state_4x"]),
        }
        action = self._read_slice(episode_id, self.action_key, anchor,
                                  anchor + self.action_window_size)[:, :self.action_dim]
        return {"obs": obs, "action": torch.from_numpy(action)}

    def get_normalizer(self, max_rows: int | None = None) -> MultiFieldNormalizer:
        fields = {"action": [], "force": [], "state": []}
        limit = max(1, int(max_rows) // len(self.episodes)) if max_rows else None
        for episode_id in range(len(self.episodes)):
            table = pq.read_table(self.episodes[episode_id][1],
                                  columns=[self.action_key, self.force_key, self.state_key])
            episode = {key: np.asarray(table[key].to_pylist(), dtype=np.float32)
                       for key in (self.action_key, self.force_key, self.state_key)}
            fields["action"].append(_subsample(episode[self.action_key][:, :self.action_dim], limit))
            fields["force"].append(_subsample(episode[self.force_key], limit))
            fields["state"].append(_subsample(episode[self.state_key], limit))
        normalizer = MultiFieldNormalizer()
        for name, chunks in fields.items():
            normalizer[name] = FieldNormalizer.from_data_limits(np.concatenate(chunks))
        normalizer["force_4x"] = FieldNormalizer.from_data_limits(np.concatenate(fields["force"]))
        normalizer["state_4x"] = FieldNormalizer.from_data_limits(np.concatenate(fields["state"]))
        return normalizer

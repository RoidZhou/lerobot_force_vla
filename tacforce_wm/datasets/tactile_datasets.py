from __future__ import annotations

import glob
import os
import re
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from utils.normalizer import FieldNormalizer, MultiFieldNormalizer


_EPISODE_RE = re.compile(r"episode_(\d+)\.parquet$")


def _subsample_rows(arr: np.ndarray, max_rows: int | None) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1, arr.shape[-1])
    if max_rows is None or max_rows <= 0 or len(flat) <= max_rows:
        return flat
    indices = np.linspace(0, len(flat) - 1, num=max_rows, dtype=np.int64)
    return flat[indices]


def _resize_tactile_rows(tactile: np.ndarray, target_rows: int = 35) -> np.ndarray:
    """Map [T, 20, 20, C] to [T, 35, 20, C] with linear row interpolation."""
    source_rows = tactile.shape[1]
    positions = np.linspace(0, source_rows - 1, target_rows, dtype=np.float32)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, source_rows - 1)
    weight = (positions - lower).reshape(1, target_rows, 1, 1)
    return tactile[:, lower] * (1.0 - weight) + tactile[:, upper] * weight


def _pack_tactile(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape or left.ndim != 3 or left.shape[1:] != (400, 3):
        raise ValueError(
            "Expected left/right tactile arrays with matching [T, 400, 3] shapes, "
            f"got {left.shape} and {right.shape}."
        )
    left = _resize_tactile_rows(left.reshape(-1, 20, 20, 3))
    right = _resize_tactile_rows(right.reshape(-1, 20, 20, 3))
    tactile = np.concatenate([left, right], axis=-1)
    return np.concatenate([tactile, tactile[:, -1:]], axis=1).astype(np.float32)


class TacForceDataset(Dataset):
    """Episode-safe window dataset for a local LeRobot v2.1 parquet dataset."""

    def __init__(
        self,
        root_dir: str,
        training: bool = True,
        window_size: int | None = 16,
        stride: int = 1,
        pad_to_multiple_of_4: bool = True,
        force_key: str = "observation.force_torque",
        state_key: str = "observation.state",
        tactile_left_key: str = "observation.tactile_left.displacement",
        tactile_right_key: str = "observation.tactile_right.displacement",
        action_key: str | None = None,
        val_fraction: float = 0.2,
        split_seed: int = 42,
        force_upsample: int = 4,
    ):
        del pad_to_multiple_of_4
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.window_size = int(window_size) if window_size is not None else None
        self.stride = max(1, int(stride))
        self.force_key = force_key
        self.state_key = state_key
        self.tactile_left_key = tactile_left_key
        self.tactile_right_key = tactile_right_key
        self.action_key = action_key
        self.force_upsample = int(force_upsample)
        if self.force_upsample != 4:
            raise ValueError("The unchanged condition encoder requires force_upsample=4.")

        paths = sorted(glob.glob(os.path.join(self.root_dir, "data", "chunk-*", "episode_*.parquet")))
        if not paths:
            raise FileNotFoundError(f"No LeRobot episode parquet files found under {self.root_dir}/data")

        episodes = []
        for path in paths:
            match = _EPISODE_RE.search(path)
            if match:
                episodes.append((int(match.group(1)), path))
        rng = np.random.default_rng(int(split_seed))
        order = rng.permutation(len(episodes))
        val_count = max(1, int(round(len(episodes) * float(val_fraction)))) if len(episodes) > 1 else 0
        val_indices = set(order[-val_count:].tolist()) if val_count else set()
        selected = [ep for i, ep in enumerate(episodes) if (i not in val_indices) == training]
        if not selected:
            raise ValueError("The requested train/validation split contains no episodes.")
        self.episodes = selected

        self.windows: list[tuple[int, int, int]] = []
        for local_episode, (_, path) in enumerate(self.episodes):
            length = pq.ParquetFile(path).metadata.num_rows
            size = length if self.window_size is None else self.window_size
            for start in range(0, length - size + 1, self.stride):
                self.windows.append((local_episode, start, start + size))
        if not self.windows:
            raise ValueError(f"No windows of size {self.window_size} found in selected episodes.")

    @property
    def _columns(self) -> list[str]:
        columns = [self.tactile_left_key, self.tactile_right_key, self.force_key, self.state_key]
        if self.action_key:
            columns.append(self.action_key)
        return columns

    @lru_cache(maxsize=4)
    def _load_episode(self, local_episode: int) -> dict[str, np.ndarray]:
        table = pq.read_table(self.episodes[local_episode][1], columns=self._columns)
        return {key: np.asarray(table[key].to_pylist(), dtype=np.float32) for key in self._columns}

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        local_episode, start, end = self.windows[idx]
        episode = self._load_episode(local_episode)
        tactile = _pack_tactile(
            episode[self.tactile_left_key][start:end],
            episode[self.tactile_right_key][start:end],
        )
        # The LeRobot streams are synchronous; repetition preserves the original
        # world model's four-times-faster conditioning interface.
        force_4x = np.repeat(episode[self.force_key][start:end], self.force_upsample, axis=0)
        state_4x = np.repeat(episode[self.state_key][start:end], self.force_upsample, axis=0)
        result = {
            "tactile": torch.from_numpy(tactile),
            "force_4x": torch.from_numpy(force_4x),
            "state_4x": torch.from_numpy(state_4x),
        }
        if self.action_key:
            result["action"] = torch.from_numpy(episode[self.action_key][start:end])
        return result

    def get_normalizer(self, max_rows: int | None = None) -> MultiFieldNormalizer:
        tactile_rows, force_rows, state_rows = [], [], []
        per_episode_limit = None
        if max_rows and max_rows > 0:
            per_episode_limit = max(1, int(max_rows) // len(self.episodes))
        for local_episode in range(len(self.episodes)):
            episode = self._load_episode(local_episode)
            tactile_rows.append(_subsample_rows(_pack_tactile(
                episode[self.tactile_left_key], episode[self.tactile_right_key]
            ), per_episode_limit))
            force_rows.append(_subsample_rows(episode[self.force_key], per_episode_limit))
            state_rows.append(_subsample_rows(episode[self.state_key], per_episode_limit))

        normalizer = MultiFieldNormalizer()
        normalizer["tactile"] = FieldNormalizer.from_data_limits(np.concatenate(tactile_rows))
        normalizer["force_4x"] = FieldNormalizer.from_data_limits(np.concatenate(force_rows))
        normalizer["state_4x"] = FieldNormalizer.from_data_limits(np.concatenate(state_rows))
        return normalizer

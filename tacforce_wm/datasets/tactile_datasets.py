from __future__ import annotations

import glob
import os
import re
import time

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
        preload_to_ram: bool = True,
    ):
        del pad_to_multiple_of_4
        self.training = bool(training)
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.window_size = int(window_size) if window_size is not None else None
        self.stride = max(1, int(stride))
        self.force_key = force_key
        self.state_key = state_key
        self.tactile_left_key = tactile_left_key
        self.tactile_right_key = tactile_right_key
        self.action_key = action_key
        self.force_upsample = int(force_upsample)
        self.preload_to_ram = bool(preload_to_ram)
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

        if not self.preload_to_ram:
            raise ValueError(
                "LeRobot dynamics training requires preload_to_ram=true to avoid "
                "random Parquet episode re-decoding."
            )

        self.episode_data = self._preload_episodes()

        self.windows: list[tuple[int, int, int]] = []
        for local_episode, episode in enumerate(self.episode_data):
            length = episode["tactile"].shape[0]
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

    def _preload_episodes(self) -> list[dict[str, np.ndarray]]:
        started = time.perf_counter()
        episode_data: list[dict[str, np.ndarray]] = []
        total_bytes = 0
        split_name = "train" if self.training else "validation"
        print(f"[TacForceDataset] Preloading {len(self.episodes)} {split_name} episodes into RAM...")
        for position, (episode_index, path) in enumerate(self.episodes, start=1):
            table = pq.read_table(path, columns=self._columns)
            arrays = {
                key: np.asarray(table[key].to_pylist(), dtype=np.float32)
                for key in self._columns
            }
            # Convert spatial tactile layout exactly once per frame. Keeping only
            # the packed tensor also releases the two large raw tactile arrays.
            packed = _pack_tactile(
                arrays.pop(self.tactile_left_key),
                arrays.pop(self.tactile_right_key),
            )
            data = {
                "tactile": np.ascontiguousarray(packed),
                "force": np.ascontiguousarray(arrays.pop(self.force_key)),
                "state": np.ascontiguousarray(arrays.pop(self.state_key)),
            }
            if self.action_key:
                data["action"] = np.ascontiguousarray(arrays.pop(self.action_key))
            episode_data.append(data)
            total_bytes += sum(value.nbytes for value in data.values())
            print(
                f"[TacForceDataset] loaded {position}/{len(self.episodes)} "
                f"episode={episode_index} frames={len(packed)}",
                flush=True,
            )
        elapsed = time.perf_counter() - started
        print(
            f"[TacForceDataset] RAM preload complete: {total_bytes / 1024**2:.1f} MiB "
            f"in {elapsed:.1f}s",
            flush=True,
        )
        return episode_data

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        local_episode, start, end = self.windows[idx]
        episode = self.episode_data[local_episode]
        tactile = episode["tactile"][start:end]
        # The LeRobot streams are synchronous; repetition preserves the original
        # world model's four-times-faster conditioning interface.
        force_4x = np.repeat(episode["force"][start:end], self.force_upsample, axis=0)
        state_4x = np.repeat(episode["state"][start:end], self.force_upsample, axis=0)
        result = {
            "tactile": torch.from_numpy(tactile),
            "force_4x": torch.from_numpy(force_4x),
            "state_4x": torch.from_numpy(state_4x),
        }
        if self.action_key:
            result["action"] = torch.from_numpy(episode["action"][start:end])
        return result

    def get_normalizer(self, max_rows: int | None = None) -> MultiFieldNormalizer:
        tactile_rows, force_rows, state_rows = [], [], []
        per_episode_limit = None
        if max_rows and max_rows > 0:
            per_episode_limit = max(1, int(max_rows) // len(self.episodes))
        for local_episode in range(len(self.episodes)):
            episode = self.episode_data[local_episode]
            tactile_rows.append(_subsample_rows(episode["tactile"], per_episode_limit))
            force_rows.append(_subsample_rows(episode["force"], per_episode_limit))
            state_rows.append(_subsample_rows(episode["state"], per_episode_limit))

        normalizer = MultiFieldNormalizer()
        normalizer["tactile"] = FieldNormalizer.from_data_limits(np.concatenate(tactile_rows))
        normalizer["force_4x"] = FieldNormalizer.from_data_limits(np.concatenate(force_rows))
        normalizer["state_4x"] = FieldNormalizer.from_data_limits(np.concatenate(state_rows))
        return normalizer

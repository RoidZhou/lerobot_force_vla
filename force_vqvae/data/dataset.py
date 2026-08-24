"""LeRobot force/torque window dataset for ForceVQVAE pretraining."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from .stats import ForceStats, resolve_force_key


class ForceWindowDataset(Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | None = None,
        force_key: str = "force_torque",
        force_dim: int = 6,
        window: int = 16,
        stride: int = 1,
        episodes: Optional[List[int]] = None,
        stats: Optional[ForceStats] = None,
    ):
        self.repo_id = repo_id
        self.root = root
        self.force_key = force_key
        self.force_dim = int(force_dim)
        self.window = int(window)
        self.stride = max(1, int(stride))
        self.ds = LeRobotDataset(repo_id=repo_id, root=root, episodes=episodes, download_videos=False)
        self.force_key = resolve_force_key(self.ds, force_key)

        if self.force_key not in self.ds.hf_dataset.column_names:
            raise KeyError(
                f"`{force_key}` resolved to `{self.force_key}`, but it is not in parquet columns. "
                f"Available columns: {self.ds.hf_dataset.column_names}"
            )
        self.stats = stats if stats is not None else ForceStats.from_lerobot_dataset(
            self.ds, force_key=self.force_key, force_dim=force_dim
        )

        episode_ids = torch.stack(self.ds.hf_dataset["episode_index"]).numpy().astype(np.int64)
        self._episode_ranges = {}
        for ep_idx in np.unique(episode_ids):
            positions = np.nonzero(episode_ids == ep_idx)[0]
            self._episode_ranges[int(ep_idx)] = (int(positions[0]), int(positions[-1]) + 1)

        self._episode_indices = sorted(self._episode_ranges)

        self._windows: List[Tuple[int, int, int]] = []
        for ep_idx in self._episode_indices:
            ep_start, ep_end = self._episode_ranges[ep_idx]
            ep_len = ep_end - ep_start
            if ep_len < self.window:
                continue
            for rel_start in range(0, ep_len - self.window + 1, self.stride):
                self._windows.append((ep_idx, ep_start + rel_start, rel_start))

        if not self._windows:
            raise RuntimeError(f"No force windows found with window={self.window}, stride={self.stride}.")

        self._cache_ep_idx = -1
        self._cache_force = None

    def __len__(self) -> int:
        return len(self._windows)

    @property
    def num_episodes(self) -> int:
        return len(self._episode_indices)

    def _load_episode_force(self, ep_idx: int) -> np.ndarray:
        if ep_idx == self._cache_ep_idx and self._cache_force is not None:
            return self._cache_force

        ep_start, ep_end = self._episode_ranges[ep_idx]
        values = []
        for idx in range(ep_start, ep_end):
            value = self.ds.hf_dataset[idx][self.force_key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            values.append(np.asarray(value, dtype=np.float32).reshape(-1))
        force = np.stack(values, axis=0)
        if force.shape[1] != self.force_dim:
            raise ValueError(f"Expected `{self.force_key}` dim {self.force_dim}, got {force.shape[1]}.")

        self._cache_ep_idx = ep_idx
        self._cache_force = force
        return force

    def __getitem__(self, idx: int) -> Dict:
        ep_idx, abs_start, rel_start = self._windows[idx]
        force = self._load_episode_force(ep_idx)
        window = force[rel_start : rel_start + self.window]
        window_norm = self.stats.normalize(window).astype(np.float32, copy=False)
        magnitude = float(np.linalg.norm(window))

        return {
            "force": torch.from_numpy(window_norm),
            "magnitude": torch.tensor(magnitude, dtype=torch.float32),
            "episode_index": torch.tensor(ep_idx, dtype=torch.long),
            "frame_index": torch.tensor(abs_start, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        return {
            "force": torch.stack([b["force"] for b in batch], dim=0),
            "magnitude": torch.stack([b["magnitude"] for b in batch], dim=0),
            "episode_index": torch.stack([b["episode_index"] for b in batch], dim=0),
            "frame_index": torch.stack([b["frame_index"] for b in batch], dim=0),
        }


def split_episode_indices(num_episodes: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if num_episodes < 2:
        raise RuntimeError(f"Need at least 2 episodes for train/val split, got {num_episodes}.")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_episodes)
    n_val = max(1, int(round(num_episodes * val_ratio)))
    val = sorted(int(i) for i in perm[:n_val])
    train = sorted(int(i) for i in perm[n_val:])
    return train, val


def build_train_val_datasets(
    repo_id: str,
    root: str | None = None,
    force_key: str = "force_torque",
    force_dim: int = 6,
    window: int = 16,
    stride: int = 1,
    val_ratio: float = 0.02,
    seed: int = 42,
    stats: Optional[ForceStats] = None,
) -> tuple[ForceWindowDataset, ForceWindowDataset, ForceStats]:
    full = LeRobotDataset(repo_id=repo_id, root=root, download_videos=False)
    resolved_force_key = resolve_force_key(full, force_key)
    if stats is None:
        stats = ForceStats.from_lerobot_dataset(full, force_key=resolved_force_key, force_dim=force_dim)
    train_eps, val_eps = split_episode_indices(full.meta.total_episodes, val_ratio=val_ratio, seed=seed)

    train_ds = ForceWindowDataset(
        repo_id=repo_id,
        root=root,
        force_key=resolved_force_key,
        force_dim=force_dim,
        window=window,
        stride=stride,
        episodes=train_eps,
        stats=stats,
    )
    val_ds = ForceWindowDataset(
        repo_id=repo_id,
        root=root,
        force_key=resolved_force_key,
        force_dim=force_dim,
        window=window,
        stride=stride,
        episodes=val_eps,
        stats=stats,
    )
    return train_ds, val_ds, stats

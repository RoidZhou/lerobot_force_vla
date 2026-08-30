from __future__ import annotations

import numpy as np
import torch


def normalize_vector(v: np.ndarray) -> np.ndarray:
    v_mag = np.linalg.norm(v, axis=1, keepdims=True)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag


def ortho6d_to_rotation_matrix(ortho6d: np.ndarray) -> np.ndarray:
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]
    x = normalize_vector(x_raw)
    z = np.cross(x, y_raw)
    z = normalize_vector(z)
    y = np.cross(z, x)

    x = x[:, :, np.newaxis]
    y = y[:, :, np.newaxis]
    z = z[:, :, np.newaxis]
    return np.concatenate((x, y, z), axis=2)


def pose_3d_9d_to_homo_matrix_batch(pose: np.ndarray) -> np.ndarray:
    if pose.shape[1] not in (3, 9):
        raise ValueError(f"pose should be (N, 3) or (N, 9), got {pose.shape}")
    mat = np.eye(4)[None, :, :].repeat(pose.shape[0], axis=0)
    mat[:, :3, 3] = pose[:, :3]
    if pose.shape[1] == 9:
        mat[:, :3, :3] = ortho6d_to_rotation_matrix(pose[:, 3:9])
    return mat


def homo_matrix_to_pose_9d_batch(mat: np.ndarray) -> np.ndarray:
    if mat.shape[1:] != (4, 4):
        raise ValueError(f"mat should be (N, 4, 4), got {mat.shape}")
    pose = np.zeros((mat.shape[0], 9), dtype=mat.dtype)
    pose[:, :3] = mat[:, :3, 3]
    pose[:, 3:9] = mat[:, :3, :2].swapaxes(1, 2).reshape(mat.shape[0], -1)
    return pose


def _tcp_dim_list_by_action_dim(d: int) -> list[np.ndarray]:
    if d in (3, 4):
        return [np.arange(3)]
    if d in (6, 8):
        return [np.arange(3), np.arange(3, 6)]
    if d in (9, 10):
        return [np.arange(9)]
    if d in (18, 20):
        return [np.arange(9), np.arange(9, 18)]
    raise NotImplementedError(f"Unsupported action dim: {d}")


def absolute_actions_to_relative_actions(
    actions: np.ndarray,
    base_absolute_action: np.ndarray | None = None,
) -> np.ndarray:
    actions = actions.copy()
    _, d = actions.shape
    tcp_dim_list = _tcp_dim_list_by_action_dim(d)

    if base_absolute_action is None:
        base_absolute_action = actions[0].copy()
    for tcp_dim in tcp_dim_list:
        base_tcp_pose_mat = pose_3d_9d_to_homo_matrix_batch(base_absolute_action[None, tcp_dim])
        actions[:, tcp_dim] = homo_matrix_to_pose_9d_batch(
            np.linalg.inv(base_tcp_pose_mat) @ pose_3d_9d_to_homo_matrix_batch(actions[:, tcp_dim])
        )[:, : len(tcp_dim)]
    return actions


def absolute_actions_to_relative_tensor(actions: torch.Tensor, base_actions: torch.Tensor) -> torch.Tensor:
    return _convert_action_tensor(actions, base_actions, inverse=True)


def relative_actions_to_absolute_tensor(actions: torch.Tensor, base_actions: torch.Tensor) -> torch.Tensor:
    return _convert_action_tensor(actions, base_actions, inverse=False)


def action_base_from_state(state_hist: torch.Tensor, action_dim: int) -> torch.Tensor:
    return state_hist[..., -1, :action_dim]


def _normalize_vector_tensor(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-8)


def _ortho6d_to_rotation_matrix_tensor(ortho6d: torch.Tensor) -> torch.Tensor:
    x_raw = ortho6d[..., 0:3]
    y_raw = ortho6d[..., 3:6]
    x = _normalize_vector_tensor(x_raw)
    z = _normalize_vector_tensor(torch.cross(x, y_raw, dim=-1))
    y = torch.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def _pose_3d_9d_to_homo_matrix_tensor(pose: torch.Tensor) -> torch.Tensor:
    pose_dim = int(pose.shape[-1])
    if pose_dim not in (3, 9):
        raise ValueError(f"pose should end with 3 or 9 dims, got {pose.shape}")

    mat = torch.eye(4, device=pose.device, dtype=pose.dtype).expand(*pose.shape[:-1], 4, 4).clone()
    mat[..., :3, 3] = pose[..., :3]
    if pose_dim == 9:
        mat[..., :3, :3] = _ortho6d_to_rotation_matrix_tensor(pose[..., 3:9])
    return mat


def _homo_matrix_to_pose_9d_tensor(mat: torch.Tensor) -> torch.Tensor:
    if tuple(mat.shape[-2:]) != (4, 4):
        raise ValueError(f"mat should end with shape (4, 4), got {mat.shape}")
    pose = torch.zeros(*mat.shape[:-2], 9, device=mat.device, dtype=mat.dtype)
    pose[..., :3] = mat[..., :3, 3]
    pose[..., 3:9] = mat[..., :3, :2].transpose(-1, -2).reshape(*mat.shape[:-2], 6)
    return pose


def _tcp_slices_by_action_dim(d: int) -> list[slice]:
    if d in (3, 4):
        return [slice(0, 3)]
    if d in (6, 8):
        return [slice(0, 3), slice(3, 6)]
    if d in (9, 10):
        return [slice(0, 9)]
    if d in (18, 20):
        return [slice(0, 9), slice(9, 18)]
    raise NotImplementedError(f"Unsupported action dim: {d}")


def _convert_action_tensor(actions: torch.Tensor, base_actions: torch.Tensor, *, inverse: bool) -> torch.Tensor:
    action_dim = int(actions.shape[-1])
    if int(base_actions.shape[-1]) != action_dim or actions.numel() != base_actions.numel():
        raise ValueError(
            "actions and base_actions must have matching leading dimensions, "
            f"got {actions.shape} and {base_actions.shape}"
        )

    original_shape = actions.shape
    original_dtype = actions.dtype
    work_dtype = torch.float32 if original_dtype != torch.float64 else torch.float64
    out = actions.reshape(-1, action_dim).to(dtype=work_dtype).clone()
    base = base_actions.reshape(-1, action_dim).to(device=actions.device, dtype=work_dtype)

    for tcp_slice in _tcp_slices_by_action_dim(action_dim):
        pose_dim = tcp_slice.stop - tcp_slice.start
        base_mat = _pose_3d_9d_to_homo_matrix_tensor(base[..., tcp_slice])
        action_mat = _pose_3d_9d_to_homo_matrix_tensor(out[..., tcp_slice])
        if inverse:
            converted_mat = torch.linalg.inv(base_mat) @ action_mat
        else:
            converted_mat = base_mat @ action_mat
        out[..., tcp_slice] = _homo_matrix_to_pose_9d_tensor(converted_mat)[..., :pose_dim]

    return out.reshape(original_shape).to(dtype=original_dtype)

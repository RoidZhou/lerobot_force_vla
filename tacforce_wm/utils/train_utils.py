from __future__ import annotations

import os
import random
import re
from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
import yaml


_PLACEHOLDER = re.compile(r"\$\{([^}]+)\}")
_MISSING = object()
_SUPPORTED_TASKS = {"policy", "dynamics"}
_SECTION_KEYS = ("runtime", "output", "data", "model", "train", "checkpoint")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _get_by_path(cfg: Mapping[str, Any], path: str, default: Any = _MISSING) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            if default is _MISSING:
                raise KeyError(path)
            return default
        cur = cur[part]
    return cur


def _set_by_path(cfg: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def cfg_get(cfg: Mapping[str, Any], path: str, default: Any = None) -> Any:
    return _get_by_path(cfg, path, default)


def _expand_cfg_strings_inplace(cfg: dict) -> None:
    def expand_str(s: str) -> str:
        def repl(m: re.Match[str]) -> str:
            key = m.group(1).strip()
            value = _get_by_path(cfg, key, _MISSING)
            if value is _MISSING:
                raise KeyError(f"Unknown config placeholder '${{{key}}}'")
            if isinstance(value, (dict, list)) or value is None:
                raise TypeError(
                    f"Placeholder '${{{key}}}' must resolve to scalar, got {type(value)}"
                )
            return str(value)

        return _PLACEHOLDER.sub(repl, s)

    def walk(x: Any):
        if isinstance(x, str) and "${" in x:
            return expand_str(x)
        if isinstance(x, dict):
            for k in list(x.keys()):
                x[k] = walk(x[k])
            return x
        if isinstance(x, list):
            for i in range(len(x)):
                x[i] = walk(x[i])
            return x
        return x

    walk(cfg)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Config at {path} must be a YAML mapping, got {type(cfg).__name__}.")
    return cfg


def _validate_mapping_section(cfg: Mapping[str, Any], path: str) -> None:
    value = _get_by_path(cfg, path, _MISSING)
    if value is _MISSING:
        raise KeyError(f"Missing required config section '{path}'.")
    if not isinstance(value, Mapping):
        raise TypeError(f"Config section '{path}' must be a mapping, got {type(value).__name__}.")


def _validate_canonical_config(task: str, cfg: Mapping[str, Any]) -> None:
    if task not in _SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task '{task}'. Expected one of {sorted(_SUPPORTED_TASKS)}.")

    cfg_task = str(cfg.get("task") or "").lower()
    if cfg_task != task:
        raise ValueError(f"Config task mismatch: expected '{task}', got '{cfg_task or None}'.")

    for key in ("seed", "task", *_SECTION_KEYS):
        if key not in cfg:
            raise KeyError(f"Missing required top-level config key '{key}'.")

    for section in _SECTION_KEYS:
        _validate_mapping_section(cfg, section)

    _validate_mapping_section(cfg, "train.optimizer")
    _validate_mapping_section(cfg, "model.backend")
    _validate_mapping_section(cfg, f"model.{task}")

    for path in (
        "runtime.device",
        "output.root_dir",
        "output.run_name",
        "data.root_dir",
        "data.window_size",
        "train.batch_size",
        "train.val_batch_size",
        "train.epochs",
        "checkpoint.save_every",
        "checkpoint.val_every",
        "checkpoint.best_metric",
    ):
        _get_by_path(cfg, path)


def _apply_runtime_overrides(cfg: dict[str, Any], overrides: Mapping[str, Any] | None) -> None:
    if not overrides:
        return
    for path, value in overrides.items():
        if value is not None:
            _set_by_path(cfg, path, value)


def build_canonical_config(task: str, source: str | Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(source, str):
        raw = load_yaml(source)
    else:
        raw = deepcopy(source)

    if not isinstance(raw, Mapping):
        raise TypeError(f"Config source must be a mapping, got {type(raw).__name__}.")

    task = str(task or raw.get("task") or "").lower()
    if not task:
        raise ValueError("Task must be provided when building canonical config.")
    if task not in _SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task '{task}'. Expected one of {sorted(_SUPPORTED_TASKS)}.")

    raw_task = str(raw.get("task") or "").lower()
    if raw_task and raw_task != task:
        raise ValueError(f"Config task mismatch: expected '{task}', got '{raw_task}'.")

    cfg = deepcopy(dict(raw))
    _apply_runtime_overrides(cfg, overrides)
    _expand_cfg_strings_inplace(cfg)
    _validate_canonical_config(task, cfg)
    return cfg


def _config_items(cfg: Any):
    if isinstance(cfg, Mapping):
        return cfg.items()
    return vars(cfg).items()


def log_hparams_to_tensorboard(writer, cfg: Any, log_dir: str) -> None:
    if writer is None:
        return

    writer.add_text("meta/log_dir", log_dir, 0)
    for k, v in _config_items(cfg):
        if isinstance(v, (dict, list)):
            text = yaml.safe_dump(v, sort_keys=False)
        else:
            text = str(v)
        writer.add_text(f"hparams/{k}", text, 0)


def init_csv_log(log_dir: str, filename: str = "loss_log.csv") -> str:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("epoch,split,metric_name,metric_value\n")
    return log_path


def append_metric_lines(log_path: str, epoch: int, split: str, metric_dict: Mapping[str, Any]) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        for k, v in metric_dict.items():
            f.write(f"{epoch},{split},{k},{float(v):.8f}\n")


def append_text_log(log_path: str, message: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def detach_scalar_dict(d: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            if v.numel() == 1:
                out[k] = v.detach().item()
            else:
                out[k] = v.detach().float().mean().item()
        else:
            out[k] = float(v)
    return out


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return (not is_dist_initialized()) or dist.get_rank() == 0


def reduce_dict_across_processes(metric_sum: Mapping[str, float], count: float):
    if not is_dist_initialized():
        return dict(metric_sum), count

    keys = sorted(metric_sum.keys())
    values = [float(metric_sum[k]) for k in keys]
    values.append(float(count))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    reduced = {k: tensor[i].item() for i, k in enumerate(keys)}
    reduced_count = tensor[-1].item()
    return reduced, reduced_count


def move_batch_to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def normalize_batch_dict(batch: dict, normalizer, keys=None) -> dict:
    out = {}
    for k, v in batch.items():
        if keys is not None and k not in keys:
            out[k] = v
            continue
        out[k] = normalizer[k].normalize(v)
    return out


def preprocess_batch(batch: dict, normalizer, device, normalize_keys=None) -> dict:
    batch = move_batch_to_device(batch, device)
    batch = normalize_batch_dict(batch, normalizer, keys=normalize_keys)
    return batch


def get_batch_size(batch: dict) -> int:
    for v in batch.values():
        if torch.is_tensor(v):
            return int(v.size(0))
    return 1


def is_distributed_launch() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size > 1 or "LOCAL_RANK" in os.environ


def sync_module_gradients(module) -> None:
    if not is_dist_initialized():
        return

    world_size = dist.get_world_size()
    if world_size <= 1:
        return

    for param in module.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size

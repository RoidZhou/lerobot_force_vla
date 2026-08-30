from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.train_utils import cfg_get, detach_scalar_dict, log_hparams_to_tensorboard, set_seed
from datasets.policy_lerobot_dataset import LeRobotPolicyDataset
from models.dynamics import build_dynamics
from models.policy.flow_policy import FlowMatchingPolicy
from utils.tensor import move_to_device


def get_autocast_context(device: torch.device, use_amp: bool):
    enabled = bool(use_amp and device.type == "cuda")
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def build_dataloaders(cfg: dict):
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    dataset_kwargs = dict(data_cfg)
    if dataset_kwargs.get("n_image_steps") is None:
        dataset_kwargs["n_image_steps"] = cfg_get(cfg, "model.policy.curr_steps", 1)
    if dataset_kwargs.get("action_representation") is None:
        dataset_kwargs["action_representation"] = cfg_get(cfg, "model.policy.action_representation", "absolute")
    num_workers = int(train_cfg.get("num_workers", 8))
    pin_memory = bool(train_cfg.get("pin_memory", True))
    persistent_workers = bool(train_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(train_cfg.get("prefetch_factor", 4))

    train_ds = LeRobotPolicyDataset(split="train", **dataset_kwargs)
    val_ds = LeRobotPolicyDataset(split="val", **dataset_kwargs)

    train_loader_kwargs = {
        "batch_size": train_cfg.get("batch_size", 32),
        "shuffle": True,
        "num_workers": num_workers,
        "drop_last": True,
        "pin_memory": pin_memory,
    }
    val_loader_kwargs = {
        "batch_size": train_cfg.get("val_batch_size", train_cfg.get("batch_size", 32)),
        "shuffle": False,
        "num_workers": num_workers,
        "drop_last": False,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = persistent_workers
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        val_loader_kwargs["persistent_workers"] = persistent_workers
        val_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, **val_loader_kwargs)
    return train_ds, val_ds, train_loader, val_loader


def build_policy(
    cfg: dict,
    device: torch.device,
    train_dataset: LeRobotPolicyDataset = None,
    dynamics_bundle: Optional[dict] = None,
):
    latent_dim = getattr(train_dataset, "cached_latent_dim", None)
    if getattr(train_dataset, "cached_image_backbone_feat", None) is not None and not cfg["model"]["policy"].get(
        "freeze_image_encoder", True
    ):
        raise ValueError("Cached image_backbone_feat requires policy.freeze_image_encoder=True.")
    dynamics = build_dynamics(cfg["model"]["backend"], device=str(device), bundle=dynamics_bundle)

    policy_cfg = cfg["model"]["policy"]
    policy = FlowMatchingPolicy(
        dynamics=dynamics,
        latent_dim=latent_dim,
        action_dim=policy_cfg.get("action_dim", 10),
        force_dim=policy_cfg.get("force_dim", 6),
        state_dim=policy_cfg.get("state_dim", 9),
        cond_dim=policy_cfg.get("cond_dim", 256),
        curr_steps=policy_cfg.get("curr_steps", 1),
        future_steps=policy_cfg.get("future_steps", 4),
        n_action_steps=policy_cfg.get("n_action_steps", 8),
        action_horizon=policy_cfg.get("action_horizon"),
        action_representation=policy_cfg.get("action_representation", "absolute"),
        down_dims=tuple(policy_cfg.get("down_dims", [256, 512, 1024])),
        diffusion_step_embed_dim=policy_cfg.get("diffusion_step_embed_dim", 256),
        kernel_size=policy_cfg.get("kernel_size", 5),
        n_groups=policy_cfg.get("n_groups", 8),
        image_encoder_name=policy_cfg.get("image_encoder_name", "dinov2_small"),
        freeze_image_encoder=policy_cfg.get("freeze_image_encoder", True),
        image_pretrained=policy_cfg.get("image_pretrained", True),
        dino_model_name=policy_cfg.get("dino_model_name", "vit_small_patch14_dinov2.lvd142m"),
        tactile_future_slice=policy_cfg.get("tactile_future_slice", "front"),
        tactile_cross_heads=int(policy_cfg.get("tactile_cross_heads", 2)),
    ).to(device)

    if train_dataset is not None:
        normalizer = train_dataset.get_normalizer(max_rows=cfg.get("train", {}).get("normalizer_max_rows"))
        policy.set_normalizer(normalizer)
        policy.normalizer.to(device)
    return policy


def train_one_epoch(
    policy,
    loader,
    optimizer,
    device,
    grad_clip=None,
    global_step: int = 0,
    writer=None,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
):
    policy.train()
    metric_sum = defaultdict(float)
    count = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        global_step += 1
        batch = move_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with get_autocast_context(device, use_amp):
            out = policy(batch)
            loss = out["loss"]
            scalar_metrics = detach_scalar_dict(out.get("metrics", {}))

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()

        bs = batch["action"].shape[0]
        metric_sum["loss"] += float(loss.detach().item()) * bs
        for k, v in scalar_metrics.items():
            metric_sum[k] += v * bs
        count += bs

        step_lr = optimizer.param_groups[0]["lr"]
        if writer is not None:
            writer.add_scalar("Step/lr", step_lr, global_step)
            for k, v in scalar_metrics.items():
                writer.add_scalar(f"Step/{k}", v, global_step)

        postfix = {"loss": f"{loss.detach().item():.4f}", "lr": f"{step_lr:.2e}"}
        for k, v in scalar_metrics.items():
            postfix[k] = f"{v:.4f}"
        pbar.set_postfix(postfix)

    avg = {k: v / max(count, 1) for k, v in metric_sum.items()}
    return avg, global_step


@torch.no_grad()
def validate_one_epoch(policy, loader, device, epoch: int, writer=None, use_amp: bool = False):
    policy.eval()
    metric_sum = defaultdict(float)
    count = 0

    pbar = tqdm(loader, desc="Val", leave=False)
    for batch in pbar:
        batch = move_to_device(batch, device)
        with get_autocast_context(device, use_amp):
            out = policy(batch)
            loss = out["loss"]
            scalar_metrics = detach_scalar_dict(out.get("metrics", {}))

        bs = batch["action"].shape[0]
        metric_sum["loss"] += float(loss.detach().item()) * bs
        for k, v in scalar_metrics.items():
            metric_sum[k] += v * bs
        count += bs
        pbar.set_postfix(loss=f"{loss.detach().item():.4f}")

    avg = {k: v / max(count, 1) for k, v in metric_sum.items()}
    if writer is not None:
        for k, v in avg.items():
            if k != "loss":
                writer.add_scalar(f"Epoch/val_{k}", v, epoch)
    return avg


@torch.no_grad()
def evaluate_open_loop(
    policy,
    loader,
    device,
    epoch: int,
    split: str,
    max_batches: int,
    writer=None,
):
    policy.eval()
    mse_sum = 0.0
    l1_sum = 0.0
    count = 0

    pbar = tqdm(loader, desc=f"OpenLoop-{split}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if batch_idx >= max_batches:
            break

        batch = move_to_device(batch, device)
        result = policy.predict_action(batch["obs"])
        pred_action = result.get("action_model", result["action"])
        gt_action = batch["action"]

        t = min(pred_action.shape[1], gt_action.shape[1])
        d = min(pred_action.shape[2], gt_action.shape[2])
        if t <= 0 or d <= 0:
            continue

        pred = pred_action[:, :t, :d]
        gt = gt_action[:, :t, :d]

        mse = F.mse_loss(pred, gt, reduction="mean")
        l1 = torch.mean(torch.abs(pred - gt))

        bs = gt.shape[0]
        mse_sum += float(mse.item()) * bs
        l1_sum += float(l1.item()) * bs
        count += bs

        pbar.set_postfix(mse=f"{mse.item():.4f}", l1=f"{l1.item():.4f}")

    metrics = {
        "action_mse": mse_sum / max(count, 1),
        "action_l1": l1_sum / max(count, 1),
    }

    if writer is not None:
        writer.add_scalar(f"OpenLoop/{split}_action_mse", metrics["action_mse"], epoch)
        writer.add_scalar(f"OpenLoop/{split}_action_l1", metrics["action_l1"], epoch)

    return metrics


def get_checkpoint_state(policy, _optimizer, epoch, cfg, dynamics_bundle):
    full_state = policy.state_dict()
    trainable_state = {k: v for k, v in full_state.items() if not k.startswith("dynamics.")}
    return {
        "epoch": int(epoch),
        "policy_state_dict": trainable_state,
        "normalizer_state_dict": deepcopy(policy.normalizer.state_dict()),
        "config": cfg,
        "dynamics_bundle": dynamics_bundle,
    }


def prepare_dynamics_bundle(
    policy: FlowMatchingPolicy,
    cfg: dict,
) -> Optional[dict]:
    """Build the dynamics bundle once.

    Dynamics is frozen during policy training, so wm_state_dict / normalizer
    don't change across epochs — cache them here and reuse.
    """
    dyn = getattr(policy, "dynamics", None)
    if dyn is None:
        return None
    if not hasattr(dyn, "wm"):
        raise RuntimeError("Expected FrozenTacForceDynamics with attribute 'wm'.")

    norm = getattr(dyn, "normalizer", None)
    if norm is not None and hasattr(norm, "state_dict"):
        norm_sd = deepcopy(norm.state_dict())
    else:
        norm_sd = None

    wm_dict: Optional[dict] = None
    wm_args = getattr(dyn.wm, "args", None)
    if wm_args is not None:
        wm_dict = {k: getattr(wm_args, k) for k in vars(wm_args)}

    full_state = policy.state_dict()
    wm_sd = {
        k.replace("dynamics.wm.", "", 1): v.detach().cpu().clone()
        for k, v in full_state.items()
        if k.startswith("dynamics.wm.")
    }
    return {
        "wm_config": wm_dict,
        "wm_state_dict": wm_sd,
        "normalizer_state_dict": norm_sd,
    }


def save_checkpoint(path: str, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_dataloader(loader, device: torch.device, warmup_steps: int = 10, bench_steps: int = 100):
    total_steps = warmup_steps + bench_steps
    it = iter(loader)
    timings = []
    sample_count = 0

    for step in range(total_steps):
        _sync_if_cuda(device)
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        _sync_if_cuda(device)
        dt = time.perf_counter() - t0

        if step >= warmup_steps:
            timings.append(dt)
            if isinstance(batch, dict) and "action" in batch and torch.is_tensor(batch["action"]):
                sample_count += int(batch["action"].shape[0])

    mean_step = float(np.mean(timings)) if timings else 0.0
    std_step = float(np.std(timings)) if timings else 0.0
    samples_per_sec = sample_count / max(sum(timings), 1e-12)

    print("\n=== Dataloader Benchmark ===")
    print(f"warmup={warmup_steps}, steps={bench_steps}")
    print(f"step_time_mean={mean_step * 1000:.2f} ms, step_time_std={std_step * 1000:.2f} ms")
    print(f"samples_per_sec={samples_per_sec:.2f}")
    return {
        "step_time_mean_ms": mean_step * 1000.0,
        "step_time_std_ms": std_step * 1000.0,
        "samples_per_sec": samples_per_sec,
    }


def benchmark_model_train_step(policy, loader, optimizer, device: torch.device, warmup_steps: int = 10, bench_steps: int = 100):
    total_steps = warmup_steps + bench_steps
    it = iter(loader)
    timings = []
    sample_count = 0

    policy.train()
    for step in range(total_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        batch = move_to_device(batch, device)
        _sync_if_cuda(device)
        t0 = time.perf_counter()

        out = policy(batch)
        loss = out["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        _sync_if_cuda(device)
        dt = time.perf_counter() - t0
        if step >= warmup_steps:
            timings.append(dt)
            if "action" in batch and torch.is_tensor(batch["action"]):
                sample_count += int(batch["action"].shape[0])

    mean_step = float(np.mean(timings)) if timings else 0.0
    std_step = float(np.std(timings)) if timings else 0.0
    samples_per_sec = sample_count / max(sum(timings), 1e-12)

    print("\n=== Model Train-Step Benchmark ===")
    print(f"warmup={warmup_steps}, steps={bench_steps}")
    print(f"step_time_mean={mean_step * 1000:.2f} ms, step_time_std={std_step * 1000:.2f} ms")
    print(f"samples_per_sec={samples_per_sec:.2f}")
    return {
        "step_time_mean_ms": mean_step * 1000.0,
        "step_time_std_ms": std_step * 1000.0,
        "samples_per_sec": samples_per_sec,
    }


def main(cfg: dict):
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg_get(cfg, "runtime.device", "cuda" if torch.cuda.is_available() else "cpu"))
    runtime_cfg = cfg["runtime"]
    benchmark = runtime_cfg.get("benchmark", "none")
    bench_steps = int(runtime_cfg.get("bench_steps", 100))
    bench_warmup = int(runtime_cfg.get("bench_warmup", 10))

    train_ds, val_ds, train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    policy = build_policy(cfg, device, train_ds)
    params = [p for p in policy.parameters() if p.requires_grad]
    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(train_cfg["optimizer"]["lr"]),
        weight_decay=float(train_cfg["optimizer"]["weight_decay"]),
    )

    if benchmark in {"dataloader", "both"}:
        benchmark_dataloader(train_loader, device=device, warmup_steps=bench_warmup, bench_steps=bench_steps)
    if benchmark in {"model", "both"}:
        benchmark_model_train_step(
            policy,
            train_loader,
            optimizer,
            device=device,
            warmup_steps=bench_warmup,
            bench_steps=bench_steps,
        )
    if benchmark != "none":
        print("\nBenchmark done. Skip full training loop.")
        return

    output_cfg = cfg["output"]
    output_root = output_cfg.get("root_dir", "outputs")
    run_name = output_cfg.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(run_dir, "resolved_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    writer = SummaryWriter(log_dir=run_dir)
    log_hparams_to_tensorboard(writer, cfg, run_dir)
    print(f"TensorBoard log dir: {run_dir}")

    epochs = int(train_cfg.get("epochs", 100))
    ckpt_cfg = cfg["checkpoint"]
    val_every = max(1, int(ckpt_cfg.get("val_every", 1)))
    save_every = max(1, int(ckpt_cfg.get("save_every", 5)))
    grad_clip = train_cfg.get("grad_clip", None)
    open_loop_every = int(train_cfg.get("open_loop_test_every", 0))
    open_loop_max_batches = max(1, int(train_cfg.get("open_loop_test_max_batches", 20)))
    use_amp = bool(train_cfg.get("use_amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    dynamics_bundle = prepare_dynamics_bundle(policy, cfg)

    best_val = float("inf")
    global_step = 0
    for epoch in range(1, epochs + 1):
        train_avg, global_step = train_one_epoch(
            policy,
            train_loader,
            optimizer,
            device,
            grad_clip=grad_clip,
            global_step=global_step,
            writer=writer,
            scaler=(scaler if use_amp else None),
            use_amp=use_amp,
        )
        train_loss = train_avg["loss"]

        if epoch % val_every == 0 or epoch == epochs:
            val_avg = validate_one_epoch(policy, val_loader, device, epoch=epoch, writer=writer, use_amp=use_amp)
            val_loss = val_avg["loss"]
        else:
            val_avg = None
            val_loss = None

        train_open = None
        val_open = None
        if open_loop_every > 0 and (epoch % open_loop_every == 0 or epoch == epochs):
            train_open = evaluate_open_loop(
                policy,
                train_loader,
                device,
                epoch,
                split="train",
                max_batches=open_loop_max_batches,
                writer=writer,
            )
            val_open = evaluate_open_loop(
                policy,
                val_loader,
                device,
                epoch,
                split="val",
                max_batches=open_loop_max_batches,
                writer=writer,
            )

        curr_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Epoch/lr", curr_lr, epoch)
        for k, v in train_avg.items():
            if k != "loss":
                writer.add_scalar(f"Epoch/train_{k}", v, epoch)

        msg = f"[Epoch {epoch:03d}] train_loss={train_loss:.6f}"
        if val_loss is not None:
            msg += f", val_loss={val_loss:.6f}"
        if train_open is not None:
            msg += f", train_open_mse={train_open['action_mse']:.6f}, train_open_l1={train_open['action_l1']:.6f}"
        if val_open is not None:
            msg += f", val_open_mse={val_open['action_mse']:.6f}, val_open_l1={val_open['action_l1']:.6f}"
        print(msg)

        state = get_checkpoint_state(
            policy,
            optimizer,
            epoch,
            cfg,
            dynamics_bundle,
        )
        save_checkpoint(os.path.join(ckpt_dir, "latest.pt"), state)
        if epoch % save_every == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"), state)
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), state)
            print(f"  -> Updated best checkpoint: val_loss={best_val:.6f}")

        writer.flush()

    writer.close()
    print(f"Training finished. Artifacts saved in: {run_dir}")

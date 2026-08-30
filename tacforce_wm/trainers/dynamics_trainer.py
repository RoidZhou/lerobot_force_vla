from __future__ import annotations

import os
from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.tactile_datasets import TacForceDataset
from models.dynamics.world_model import TacForceWorldModel
from utils.train_utils import (
    append_metric_lines,
    append_text_log,
    detach_scalar_dict,
    get_batch_size,
    init_csv_log,
    is_distributed_launch,
    is_main_process,
    log_hparams_to_tensorboard,
    preprocess_batch,
    reduce_dict_across_processes,
    set_seed,
)
from utils.normalizer import MultiFieldNormalizer


def build_model(cfg: dict):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]["dynamics"]
    return TacForceWorldModel(SimpleNamespace(**{**model_cfg, "window_size": data_cfg["window_size"]}))


def build_dataset(cfg: dict, training: bool = True):
    data_cfg = cfg["data"]
    keys_cfg = data_cfg["keys"]
    return TacForceDataset(
        root_dir=data_cfg["root_dir"],
        training=training,
        window_size=data_cfg.get("window_size"),
        stride=data_cfg.get("stride", 1),
        pad_to_multiple_of_4=data_cfg.get("pad_to_multiple_of_4", True),
        force_key=keys_cfg.get("force", "left_wrist_force"),
        state_key=keys_cfg.get("state", "left_robot_tcp_pose"),
        tactile_left_key=keys_cfg.get("tactile_left", "left_gripper1_tactile"),
        tactile_right_key=keys_cfg.get("tactile_right", "left_gripper2_tactile"),
        action_key=keys_cfg.get("action"),
        val_fraction=data_cfg.get("val_fraction", 0.2),
        split_seed=data_cfg.get("split_seed", cfg.get("seed", 42)),
        force_upsample=data_cfg.get("force_upsample", 4),
    )


def build_normalizer(train_dataset: TacForceDataset, device, max_rows: int | None = None):
    normalizer = train_dataset.get_normalizer(max_rows=max_rows)
    normalizer.to(device)
    return normalizer


def save_dynamics_checkpoint(path: str, model_state_dict: dict, normalizer_state_dict: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model_state_dict, "normalizer": normalizer_state_dict}, path)


def load_dynamics_checkpoint(path: str, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if not (isinstance(ckpt, dict) and "model" in ckpt):
        raise KeyError(f"Dynamics checkpoint {path} is missing 'model' field.")
    return ckpt


def get_autocast_context(device: torch.device, use_amp: bool):
    enabled = bool(use_amp and device.type == "cuda")
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    normalizer,
    normalize_keys,
    epoch: int,
    global_step: int,
    writer=None,
    lr_warmup_steps: int = 1000,
    base_lr: float = 3e-4,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
):
    model.train()
    metric_sum = defaultdict(float)
    sample_count = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]") if is_main_process() else loader
    for batch in pbar:
        global_step += 1
        if global_step <= lr_warmup_steps:
            curr_lr = base_lr * global_step / max(1, lr_warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = curr_lr

        batch = preprocess_batch(batch, normalizer, device, normalize_keys)

        optimizer.zero_grad(set_to_none=True)
        with get_autocast_context(device, use_amp):
            out = model(batch)
            loss = out["loss"]
            scalar_metrics = detach_scalar_dict({k: v for k, v in out.items() if k != "loss"})

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = get_batch_size(batch)
        sample_count += bs
        metric_sum["loss"] += loss.detach().item() * bs
        for k, v in scalar_metrics.items():
            metric_sum[k] += v * bs

        if is_main_process():
            step_lr = optimizer.param_groups[0]["lr"]
            if writer is not None:
                writer.add_scalar("Step/lr", step_lr, global_step)
                for k, v in scalar_metrics.items():
                    writer.add_scalar(f"Step/{k}", v, global_step)

            postfix = {"loss": f"{loss.detach().item():.4f}", "lr": f"{step_lr:.2e}"}
            for k, v in scalar_metrics.items():
                postfix[k] = f"{v:.4f}"
            pbar.set_postfix(postfix)

    metric_sum, sample_count = reduce_dict_across_processes(metric_sum, sample_count)
    avg = {k: v / max(1.0, sample_count) for k, v in metric_sum.items()}
    return avg, global_step


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    device,
    normalizer,
    normalize_keys,
    epoch: int,
    writer=None,
    use_amp: bool = False,
):
    model.eval()
    metric_sum = defaultdict(float)
    sample_count = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False) if is_main_process() else loader
    for batch in pbar:
        batch = preprocess_batch(batch, normalizer, device, normalize_keys)
        with get_autocast_context(device, use_amp):
            out = model(batch)
            loss = out["loss"]
            scalar_metrics = detach_scalar_dict({k: v for k, v in out.items() if k != "loss"})

        bs = get_batch_size(batch)
        sample_count += bs
        metric_sum["loss"] += loss.detach().item() * bs
        for k, v in scalar_metrics.items():
            metric_sum[k] += v * bs

    metric_sum, sample_count = reduce_dict_across_processes(metric_sum, sample_count)
    avg = {k: v / max(1.0, sample_count) for k, v in metric_sum.items()}
    if is_main_process() and writer is not None:
        for k, v in avg.items():
            if k != "loss":
                writer.add_scalar(f"Epoch/val_{k}", v, epoch)
    return avg


def _metric_value(metrics: dict, key: str) -> float:
    if key not in metrics:
        raise KeyError(f"Metric '{key}' not found. Available metrics: {sorted(metrics.keys())}")
    return float(metrics[key])


def train_single_process(cfg: dict):
    runtime_cfg = cfg["runtime"]
    output_cfg = cfg["output"]
    train_cfg = cfg["train"]
    ckpt_cfg = cfg["checkpoint"]
    data_cfg = cfg["data"]
    device = torch.device(runtime_cfg["device"])
    train_dataset = build_dataset(cfg, training=True)
    val_dataset = build_dataset(cfg, training=False)

    num_workers = int(train_cfg.get("num_workers", 8))
    loader_common = {
        "num_workers": num_workers,
        "pin_memory": train_cfg.get("pin_memory", True),
        "drop_last": True,
    }
    if num_workers > 0:
        loader_common["persistent_workers"] = train_cfg.get("persistent_workers", True)
        loader_common["prefetch_factor"] = train_cfg.get("prefetch_factor", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        **loader_common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["val_batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=train_cfg.get("pin_memory", True),
        persistent_workers=(train_cfg.get("persistent_workers", True) if num_workers > 0 else False),
        prefetch_factor=(train_cfg.get("prefetch_factor", 4) if num_workers > 0 else None),
    )

    normalizer = build_normalizer(train_dataset, device, max_rows=train_cfg.get("normalizer_max_rows"))
    model = build_model(cfg).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["optimizer"]["lr"]),
        weight_decay=float(train_cfg["optimizer"]["weight_decay"]),
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(train_cfg["epochs"])),
        eta_min=float(train_cfg["optimizer"]["min_lr"]),
    )

    log_dir = os.path.join(output_cfg["root_dir"], output_cfg["run_name"])
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(log_dir, "resolved_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    writer = SummaryWriter(log_dir=log_dir)
    csv_log_path = init_csv_log(log_dir, "loss_log.csv")
    txt_log_path = os.path.join(log_dir, "train_log.txt")
    log_hparams_to_tensorboard(writer, cfg, log_dir)
    print(f"TensorBoard log dir: {log_dir}")

    best_metric_name = str(ckpt_cfg.get("best_metric", "loss")).strip()
    best_metric_value = float("inf")
    global_step = 0
    normalize_keys = data_cfg.get("normalize_keys", ["tactile", "force_4x", "state_4x"])
    lr_warmup_steps = int(train_cfg.get("lr_warmup_steps", 1000))
    validate_every = max(1, int(ckpt_cfg.get("val_every", 1)))
    use_amp = bool(train_cfg.get("use_amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(int(train_cfg["epochs"])):
        train_avg, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            normalizer,
            normalize_keys,
            epoch,
            global_step,
            writer=writer,
            lr_warmup_steps=lr_warmup_steps,
            base_lr=float(train_cfg["optimizer"]["lr"]),
            scaler=(scaler if use_amp else None),
            use_amp=use_amp,
        )

        should_validate = (epoch % validate_every == 0) or (epoch == int(train_cfg["epochs"]) - 1)
        val_avg = None
        if should_validate:
            val_avg = validate_one_epoch(
                model,
                val_loader,
                device,
                normalizer,
                normalize_keys,
                epoch,
                writer,
                use_amp=use_amp,
            )

        if global_step > lr_warmup_steps:
            scheduler.step()

        curr_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Epoch/lr", curr_lr, epoch)
        for k, v in train_avg.items():
            if k != "loss":
                writer.add_scalar(f"Epoch/train_{k}", v, epoch)

        append_metric_lines(csv_log_path, epoch, "train", train_avg)
        if val_avg is not None:
            append_metric_lines(csv_log_path, epoch, "val", val_avg)

        msg = (
            f"[Epoch {epoch}] train_loss={train_avg['loss']:.6f}, "
            f"val_loss={(val_avg['loss'] if val_avg is not None else 'SKIPPED')}, lr={curr_lr:.8f}"
        )
        for k, v in train_avg.items():
            if k != "loss":
                msg += f", train_{k}={v:.6f}"
        if val_avg is not None:
            msg = msg.replace(f"val_loss={val_avg['loss']}", f"val_loss={val_avg['loss']:.6f}")
            for k, v in val_avg.items():
                if k != "loss":
                    msg += f", val_{k}={v:.6f}"
        append_text_log(txt_log_path, msg)
        print(msg)

        save_dynamics_checkpoint(os.path.join(ckpt_dir, "latest.pt"), model.state_dict(), normalizer.state_dict())
        if val_avg is not None:
            metric_value = _metric_value(val_avg, best_metric_name)
            if metric_value < best_metric_value:
                best_metric_value = metric_value
                save_dynamics_checkpoint(
                    os.path.join(ckpt_dir, "best_model.pt"),
                    model.state_dict(),
                    normalizer.state_dict(),
                )
                if best_metric_name != "loss":
                    save_dynamics_checkpoint(
                        os.path.join(ckpt_dir, f"best_model_by_{best_metric_name}.pt"),
                        model.state_dict(),
                        normalizer.state_dict(),
                    )
                append_text_log(
                    txt_log_path,
                    f"  -> best updated at epoch {epoch}, {best_metric_name}={metric_value:.6f}",
                )
        writer.flush()

    writer.close()


def train_ddp(cfg: dict):
    train_cfg = cfg["train"]
    output_cfg = cfg["output"]
    ckpt_cfg = cfg["checkpoint"]
    data_cfg = cfg["data"]
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")

    train_dataset = build_dataset(cfg, training=True)
    val_dataset = build_dataset(cfg, training=False)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    ddp_num_workers = int(train_cfg.get("ddp_num_workers", train_cfg.get("num_workers", 8)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        sampler=train_sampler,
        num_workers=ddp_num_workers,
        drop_last=True,
        pin_memory=train_cfg.get("pin_memory", True),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["val_batch_size"],
        sampler=val_sampler,
        num_workers=ddp_num_workers,
        drop_last=False,
        pin_memory=train_cfg.get("pin_memory", True),
    )

    normalizer = build_normalizer(train_dataset, device, max_rows=train_cfg.get("normalizer_max_rows"))
    model = DDP(build_model(cfg).to(device), device_ids=[local_rank], output_device=local_rank)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["optimizer"]["lr"]),
        weight_decay=float(train_cfg["optimizer"]["weight_decay"]),
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(train_cfg["epochs"])),
        eta_min=float(train_cfg["optimizer"]["min_lr"]),
    )

    writer = None
    csv_log_path = None
    txt_log_path = None
    ckpt_dir = None
    log_dir = os.path.join(output_cfg["root_dir"], output_cfg["run_name"])
    if is_main_process():
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(log_dir, "resolved_config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        writer = SummaryWriter(log_dir=log_dir)
        csv_log_path = init_csv_log(log_dir, "loss_log.csv")
        txt_log_path = os.path.join(log_dir, "train_log.txt")
        log_hparams_to_tensorboard(writer, cfg, log_dir)
        print(f"Starting DDP Training | TensorBoard log dir: {log_dir}")

    best_metric_name = str(ckpt_cfg.get("best_metric", "loss")).strip()
    best_metric_value = float("inf")
    global_step = 0
    normalize_keys = data_cfg.get("normalize_keys", ["tactile", "force_4x", "state_4x"])
    lr_warmup_steps = int(train_cfg.get("lr_warmup_steps", 1000))
    validate_every = max(1, int(ckpt_cfg.get("val_every", 1)))
    use_amp = bool(train_cfg.get("use_amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(int(train_cfg["epochs"])):
        train_sampler.set_epoch(epoch)
        train_avg, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            normalizer,
            normalize_keys,
            epoch,
            global_step,
            writer=writer,
            lr_warmup_steps=lr_warmup_steps,
            base_lr=float(train_cfg["optimizer"]["lr"]),
            scaler=(scaler if use_amp else None),
            use_amp=use_amp,
        )

        should_validate = (epoch % validate_every == 0) or (epoch == int(train_cfg["epochs"]) - 1)
        val_avg = None
        if should_validate:
            val_avg = validate_one_epoch(
                model,
                val_loader,
                device,
                normalizer,
                normalize_keys,
                epoch,
                writer,
                use_amp=use_amp,
            )

        if global_step > lr_warmup_steps:
            scheduler.step()

        if is_main_process():
            curr_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Epoch/lr", curr_lr, epoch)
            for k, v in train_avg.items():
                if k != "loss":
                    writer.add_scalar(f"Epoch/train_{k}", v, epoch)

            append_metric_lines(csv_log_path, epoch, "train", train_avg)
            if val_avg is not None:
                append_metric_lines(csv_log_path, epoch, "val", val_avg)

            msg = (
                f"[Epoch {epoch}] train_loss={train_avg['loss']:.6f}, "
                f"val_loss={(val_avg['loss'] if val_avg is not None else 'SKIPPED')}, lr={curr_lr:.8f}"
            )
            for k, v in train_avg.items():
                if k != "loss":
                    msg += f", train_{k}={v:.6f}"
            if val_avg is not None:
                msg = msg.replace(f"val_loss={val_avg['loss']}", f"val_loss={val_avg['loss']:.6f}")
                for k, v in val_avg.items():
                    if k != "loss":
                        msg += f", val_{k}={v:.6f}"

            append_text_log(txt_log_path, msg)
            print(msg)

            save_dynamics_checkpoint(os.path.join(ckpt_dir, "latest.pt"), model.module.state_dict(), normalizer.state_dict())
            if val_avg is not None:
                metric_value = _metric_value(val_avg, best_metric_name)
                if metric_value < best_metric_value:
                    best_metric_value = metric_value
                    save_dynamics_checkpoint(
                        os.path.join(ckpt_dir, "best_model.pt"),
                        model.module.state_dict(),
                        normalizer.state_dict(),
                    )
                    if best_metric_name != "loss":
                        save_dynamics_checkpoint(
                            os.path.join(ckpt_dir, f"best_model_by_{best_metric_name}.pt"),
                            model.module.state_dict(),
                            normalizer.state_dict(),
                        )
                    append_text_log(
                        txt_log_path,
                        f"  -> best updated at epoch {epoch}, {best_metric_name}={metric_value:.6f}",
                    )
            writer.flush()

    if is_main_process() and writer is not None:
        writer.close()
    dist.destroy_process_group()


@torch.no_grad()
def evaluate(cfg: dict):
    runtime_cfg = cfg["runtime"]
    train_cfg = cfg["train"]
    ckpt_cfg = cfg["checkpoint"]
    data_cfg = cfg["data"]
    device = torch.device(runtime_cfg["device"])
    val_dataset = build_dataset(cfg, training=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["val_batch_size"],
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=True,
    )

    ckpt = load_dynamics_checkpoint(ckpt_cfg["load_path"], device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    if "normalizer" not in ckpt:
        raise KeyError(
            f"Dynamics checkpoint {ckpt_cfg['load_path']} is missing 'normalizer'. "
            "Please retrain so the checkpoint embeds the normalizer."
        )
    
    normalizer = MultiFieldNormalizer()
    normalizer.load_state_dict(ckpt["normalizer"])
    normalizer.to(device)

    normalize_keys = data_cfg.get("normalize_keys", ["tactile", "force_4x", "state_4x"])
    use_amp = bool(train_cfg.get("use_amp", False) and device.type == "cuda")
    avg = validate_one_epoch(
        model,
        val_loader,
        device,
        normalizer,
        normalize_keys,
        epoch=0,
        writer=None,
        use_amp=use_amp,
    )

    print("Evaluation metrics:")
    for k, v in avg.items():
        print(f"  {k}: {v:.8f}")


def main(cfg: dict):
    set_seed(int(cfg.get("seed", 42)))
    runtime_cfg = cfg["runtime"]
    if runtime_cfg.get("test"):
        evaluate(cfg)
    elif runtime_cfg.get("ddp") or is_distributed_launch():
        train_ddp(cfg)
    else:
        train_single_process(cfg)

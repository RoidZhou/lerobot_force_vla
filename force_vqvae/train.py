"""Force VQ-VAE training loop.

Example:
    python -m lerobot.force_vqvae.train \
        --repo_id ur5_rg2_real_smolvla_dataset_force_boltnut_speed_627 \
        --data_root "/media/zhou/Elements SE/YBZHOU/ur5_rg2_real_smolvla_dataset_force_boltnut_speed_627" \
        --force_key force_torque \
        --output_dir /tmp/force_vqvae \
        --window 16 --codebook_size 256 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from lerobot.force_vqvae.data import ForceStats, build_train_val_datasets
from lerobot.force_vqvae.models import ForceVQVAE, ForceVQVAEConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--force_key", type=str, default="force_torque")
    parser.add_argument("--force_dim", type=int, default=6)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--bottleneck_channels", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_strided_blocks", type=int, default=2)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--commitment_weight", type=float, default=0.25)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--revive_freq", type=int, default=200)
    parser.add_argument("--revive_threshold", type=float, default=1.0)
    parser.add_argument("--use_magnitude_weight", type=int, default=1)
    parser.add_argument("--weight_alpha", type=float, default=2.0)
    parser.add_argument("--weight_tau", type=float, default=4.0)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=2000)
    parser.add_argument("--save_every_epoch", type=int, default=1)
    parser.add_argument("--smoke_test", type=int, default=0)
    return parser.parse_args()


def cosine_lr(step: int, total: int, warmup: int, base_lr: float, min_ratio: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cos)


def build_config(args: argparse.Namespace) -> ForceVQVAEConfig:
    return ForceVQVAEConfig(
        window=args.window,
        force_dim=args.force_dim,
        hidden_channels=args.hidden_channels,
        bottleneck_channels=args.bottleneck_channels,
        embed_dim=args.embed_dim,
        n_strided_blocks=args.n_strided_blocks,
        codebook_size=args.codebook_size,
        commitment_weight=args.commitment_weight,
        decay=args.decay,
        revive_freq=args.revive_freq,
        revive_threshold=args.revive_threshold,
        use_magnitude_weight=bool(args.use_magnitude_weight),
        weight_alpha=args.weight_alpha,
        weight_tau=args.weight_tau,
    )


def save_checkpoint(
    out_dir: str,
    model: ForceVQVAE,
    optimizer: torch.optim.Optimizer,
    stats: ForceStats,
    cfg: ForceVQVAEConfig,
    step: int,
    epoch: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": cfg.to_dict(),
        "stats": stats.to_dict(),
    }
    ckpt_path = os.path.join(out_dir, f"checkpoint_epoch{epoch:03d}.pt")
    torch.save(state, ckpt_path)
    torch.save(state, os.path.join(out_dir, "latest.pt"))
    print(f"  saved checkpoint -> {ckpt_path}")


@torch.no_grad()
def validate(model: ForceVQVAE, val_loader: DataLoader, device: torch.device, max_batches: Optional[int] = 50) -> Dict[str, float]:
    model.eval()
    sums = {"recon": 0.0, "vq": 0.0, "perp": 0.0, "active": 0.0, "n": 0}
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        force = batch["force"].to(device)
        magnitude = batch["magnitude"].to(device)
        out = model(force, magnitude)
        batch_size = force.shape[0]
        sums["recon"] += out["recon_loss"].item() * batch_size
        sums["vq"] += out["vq_loss"].item() * batch_size
        sums["perp"] += out["perplexity"].item() * batch_size
        sums["active"] += out["active_codes"].item() * batch_size
        sums["n"] += batch_size
    model.train()
    if sums["n"] == 0:
        return {"val_recon": float("nan"), "val_vq": float("nan"), "val_perplexity": float("nan"), "val_active": float("nan")}
    return {
        "val_recon": sums["recon"] / sums["n"],
        "val_vq": sums["vq"] / sums["n"],
        "val_perplexity": sums["perp"] / sums["n"],
        "val_active": sums["active"] / sums["n"],
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    run_name = args.run_name or time.strftime("force_vqvae_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"[ForceVQVAE] run_dir={run_dir}")
    print(f"[ForceVQVAE] loading dataset root={args.data_root}, repo_id={args.repo_id}, force_key={args.force_key}")
    train_ds, val_ds, stats = build_train_val_datasets(
        repo_id=args.repo_id,
        root=args.data_root,
        force_key=args.force_key,
        force_dim=args.force_dim,
        window=args.window,
        stride=args.stride,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(
        f"[ForceVQVAE] train: {train_ds.num_episodes} eps / {len(train_ds)} windows; "
        f"val: {val_ds.num_episodes} eps / {len(val_ds)} windows"
    )
    print(f"[ForceVQVAE] resolved force_key={train_ds.force_key}")
    print(f"[ForceVQVAE] force_min={stats.force_min}, force_max={stats.force_max}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=train_ds.collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=val_ds.collate_fn,
    )

    cfg = build_config(args)
    model = ForceVQVAE(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, eps=1e-8
    )
    print(f"[ForceVQVAE] config={json.dumps(cfg.to_dict(), indent=2)}")
    print(f"[ForceVQVAE] params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    global_step = 0
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            lr_now = cosine_lr(global_step, total_steps, args.warmup_steps, args.lr, args.min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = lr_now

            force = batch["force"].to(device)
            magnitude = batch["magnitude"].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(force, magnitude)
            loss = out["total_loss"]
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if global_step % args.log_every == 0:
                print(
                    f"[step {global_step:7d} | ep {epoch:3d}] "
                    f"recon={out['recon_loss'].item():.4f} "
                    f"vq={out['vq_loss'].item():.4f} "
                    f"perp={out['perplexity'].item():.1f} "
                    f"active={int(out['active_codes'].item())}/{cfg.codebook_size} "
                    f"revived={int(out['revived'].item())} "
                    f"lr={lr_now:.2e} elapsed={time.time() - start:.0f}s"
                )

            if args.val_every > 0 and global_step > 0 and global_step % args.val_every == 0:
                val = validate(model, val_loader, device)
                print("  [val] " + " ".join(f"{k}={v:.4f}" for k, v in val.items()))

            global_step += 1
            if args.smoke_test and global_step >= 5:
                break

        if args.smoke_test:
            break
        if (epoch + 1) % args.save_every_epoch == 0:
            save_checkpoint(run_dir, model, optimizer, stats, cfg, global_step, epoch)

    save_checkpoint(run_dir, model, optimizer, stats, cfg, global_step, 0 if args.smoke_test else args.epochs - 1)


if __name__ == "__main__":
    main()

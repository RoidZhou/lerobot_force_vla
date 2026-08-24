"""Encode LeRobot force/torque windows into discrete ForceVQVAE code ids."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from lerobot.force_vqvae.data import ForceStats, ForceWindowDataset
from lerobot.force_vqvae.models import ForceVQVAE, ForceVQVAEConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--force_key", type=str, default="force_torque")
    parser.add_argument("--vqvae_ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    blob = torch.load(args.vqvae_ckpt, map_location="cpu", weights_only=False)
    cfg = ForceVQVAEConfig.from_dict(blob["config"])
    stats = ForceStats.from_dict(blob["stats"])
    model = ForceVQVAE(cfg)
    model.load_state_dict(blob["model_state"])
    model.eval().to(device)

    dataset = ForceWindowDataset(
        repo_id=args.repo_id,
        root=args.data_root,
        force_key=args.force_key,
        force_dim=cfg.force_dim,
        window=cfg.window,
        stride=args.stride,
        stats=stats,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )

    codes, episode_indices, frame_indices = [], [], []
    for batch in loader:
        force = batch["force"].to(device)
        codes.append(model.encode(force).cpu())
        episode_indices.append(batch["episode_index"].cpu())
        frame_indices.append(batch["frame_index"].cpu())

    out = {
        "codes": torch.cat(codes, dim=0),
        "episode_index": torch.cat(episode_indices, dim=0),
        "frame_index": torch.cat(frame_indices, dim=0),
        "config": cfg.to_dict(),
        "stats": stats.to_dict(),
        "force_key": args.force_key,
        "stride": args.stride,
    }
    torch.save(out, args.output)
    print(f"saved {out['codes'].numel()} force codes -> {args.output}")


if __name__ == "__main__":
    main()


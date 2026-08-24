"""EMA vector quantizer with dead-code revival."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if _is_dist():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _all_gather_2d(x: torch.Tensor) -> torch.Tensor:
    if not _is_dist():
        return x
    world = dist.get_world_size()
    out = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(out, x.contiguous())
    return torch.cat(out, dim=0)


class VQEMAQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int = 1024,
        embed_dim: int = 256,
        commitment_weight: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        revive_freq: int = 200,
        revive_threshold: float = 1.0,
        init_mode: str = "uniform",
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.eps = eps
        self.revive_freq = revive_freq
        self.revive_threshold = revive_threshold

        embed = torch.empty(codebook_size, embed_dim)
        if init_mode == "kaiming":
            nn.init.kaiming_uniform_(embed, a=5**0.5)
        else:
            embed.normal_(mean=0.0, std=0.02)

        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _ema_update(self, z_e: torch.Tensor, indices: torch.Tensor) -> None:
        onehot = F.one_hot(indices.view(-1), self.codebook_size).type_as(z_e)
        local_cluster = onehot.sum(dim=0)
        local_embed = onehot.t() @ z_e

        local_cluster = _all_reduce_sum(local_cluster.contiguous())
        local_embed = _all_reduce_sum(local_embed.contiguous())

        self.cluster_size.mul_(self.decay).add_(local_cluster, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(local_embed, alpha=1 - self.decay)

        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
        self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))

    @torch.no_grad()
    def _revive_dead_codes(self, z_e_pool: torch.Tensor) -> int:
        dead = self.cluster_size < self.revive_threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0

        pool = _all_gather_2d(z_e_pool)
        if pool.shape[0] == 0:
            return 0

        if not _is_dist() or dist.get_rank() == 0:
            sel = torch.randint(0, pool.shape[0], (n_dead,), device=pool.device)
        else:
            sel = torch.zeros(n_dead, dtype=torch.long, device=pool.device)
        if _is_dist():
            dist.broadcast(sel, src=0)

        replacements = pool[sel].to(self.embed.dtype)
        dead_idx = dead.nonzero(as_tuple=False).flatten()
        self.embed[dead_idx] = replacements
        self.embed_avg[dead_idx] = replacements
        self.cluster_size[dead_idx] = self.revive_threshold * 2.0
        return n_dead

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        if z_e.dim() != 2 or z_e.shape[1] != self.embed_dim:
            raise ValueError(f"Expected z_e [B, {self.embed_dim}], got {tuple(z_e.shape)}")

        z_sq = z_e.pow(2).sum(dim=1, keepdim=True)
        e_sq = self.embed.pow(2).sum(dim=1)
        dists = z_sq + e_sq - 2.0 * (z_e @ self.embed.t())
        indices = dists.argmin(dim=1)
        z_q = self.embed[indices]

        z_q_st = z_e + (z_q - z_e).detach()
        vq_loss = self.commitment_weight * F.mse_loss(z_e, z_q.detach())

        revived = 0
        if self.training:
            self._ema_update(z_e.detach(), indices)
            self.step += 1
            if int(self.step.item()) % self.revive_freq == 0:
                revived = self._revive_dead_codes(z_e.detach())

        with torch.no_grad():
            onehot = F.one_hot(indices, self.codebook_size).type_as(z_e)
            count = _all_reduce_sum(onehot.sum(dim=0).clone())
            probs = count / (count.sum() + 1e-12)
            active = int((count > 0).sum().item())
            perplexity = torch.exp(-(probs * probs.add(1e-12).log()).sum()).item()

        info = {
            "perplexity": float(perplexity),
            "active_codes": active,
            "revived": int(revived),
        }
        return z_q_st, indices, vq_loss, info

    @torch.no_grad()
    def encode_only(self, z_e: torch.Tensor) -> torch.Tensor:
        z_sq = z_e.pow(2).sum(dim=1, keepdim=True)
        e_sq = self.embed.pow(2).sum(dim=1)
        return (z_sq + e_sq - 2.0 * (z_e @ self.embed.t())).argmin(dim=1)


"""ForceVQVAE: discrete codebook over force/torque windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .decoder import ForceDecoder
from .encoder import ForceEncoder
from .quantizer import VQEMAQuantizer


@dataclass
class ForceVQVAEConfig:
    window: int = 16
    force_dim: int = 6
    hidden_channels: int = 128
    bottleneck_channels: int = 256
    embed_dim: int = 256
    n_strided_blocks: int = 2
    codebook_size: int = 1024
    commitment_weight: float = 0.25
    decay: float = 0.99
    revive_freq: int = 200
    revive_threshold: float = 1.0
    use_magnitude_weight: bool = True
    weight_alpha: float = 2.0
    weight_tau: float = 4.0
    init_mode: str = "uniform"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict) -> "ForceVQVAEConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ForceVQVAE(nn.Module):
    def __init__(self, cfg: ForceVQVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ForceEncoder(
            window=cfg.window,
            force_dim=cfg.force_dim,
            hidden_channels=cfg.hidden_channels,
            bottleneck_channels=cfg.bottleneck_channels,
            embed_dim=cfg.embed_dim,
            n_strided_blocks=cfg.n_strided_blocks,
        )
        self.decoder = ForceDecoder(
            window=cfg.window,
            force_dim=cfg.force_dim,
            hidden_channels=cfg.hidden_channels,
            bottleneck_channels=cfg.bottleneck_channels,
            embed_dim=cfg.embed_dim,
            n_strided_blocks=cfg.n_strided_blocks,
        )
        self.quantizer = VQEMAQuantizer(
            codebook_size=cfg.codebook_size,
            embed_dim=cfg.embed_dim,
            commitment_weight=cfg.commitment_weight,
            decay=cfg.decay,
            revive_freq=cfg.revive_freq,
            revive_threshold=cfg.revive_threshold,
            init_mode=cfg.init_mode,
        )

    def _recon_weight(self, magnitude: torch.Tensor) -> torch.Tensor:
        return 1.0 + self.cfg.weight_alpha * torch.sigmoid(magnitude / self.cfg.weight_tau - 1.0)

    def forward(
        self,
        force: torch.Tensor,
        magnitude: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        force: [B, T, D] normalized force/torque window.
        magnitude: [B] raw-window L2 norm, used for contact-weighted reconstruction.
        """
        z_e = self.encoder(force)
        z_q, indices, vq_loss, qinfo = self.quantizer(z_e)
        recon = self.decoder(z_q)

        per_sample = (recon - force).pow(2).mean(dim=[1, 2])
        if self.cfg.use_magnitude_weight and magnitude is not None:
            w = self._recon_weight(magnitude.to(per_sample.device))
            recon_loss = (per_sample * w).sum() / (w.sum() + 1e-8)
        else:
            recon_loss = per_sample.mean()

        total_loss = recon_loss + vq_loss
        return {
            "recon": recon,
            "indices": indices,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "total_loss": total_loss,
            "perplexity": torch.tensor(qinfo["perplexity"], device=force.device),
            "active_codes": torch.tensor(qinfo["active_codes"], device=force.device),
            "revived": torch.tensor(qinfo["revived"], device=force.device),
            "per_sample_recon": per_sample.detach(),
        }

    @torch.no_grad()
    def encode(self, force: torch.Tensor) -> torch.Tensor:
        """force [B, T, D] -> code indices [B]."""
        return self.quantizer.encode_only(self.encoder(force))

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """indices [B] -> reconstructed normalized force [B, T, D]."""
        z_q = self.quantizer.embed[indices]
        return self.decoder(z_q)


# SIGReg adapted from le-wm:
# https://github.com/lucas-maes/le-wm/blob/main/module.py
import torch
from torch import nn
import torch.nn.functional as F

class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    Expected input:
        proj: [T, B, D]
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)

        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt

        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj: [T, B, D]
        Returns:
            scalar tensor
        """
        if proj.ndim != 3:
            raise ValueError(f"SIGReg expects [T, B, D], got {tuple(proj.shape)}")

        A = torch.randn(
            proj.size(-1), self.num_proj,
            device=proj.device,
            dtype=proj.dtype,
        )
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-8)

        x_t = (proj @ A).unsqueeze(-1) * self.t

        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)

        return statistic.mean()


class TacForceLoss(nn.Module):
    """
    pred, target: [B, H, D]
    emb: [B, T, D]
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        sigreg_weight: float = 0.001,
        sigreg_knots: int = 17,
        sigreg_num_proj: int = 256,
        delta_latent_weight: float = 0.0,
    ):
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.sigreg_weight = float(sigreg_weight)
        self.delta_latent_weight = float(delta_latent_weight)

        self.sigreg = SIGReg(
            knots=sigreg_knots,
            num_proj=sigreg_num_proj,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor, emb=None):
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have same shape, got {tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        if pred.ndim != 3:
            raise ValueError(f"TacForceLoss expects pred/target [B, H, D], got {tuple(pred.shape)}")

        loss_mse = F.mse_loss(pred, target)

        if self.delta_latent_weight > 0 and pred.size(1) > 1:
            d_pred = pred[:, 1:] - pred[:, :-1]
            d_tgt = target[:, 1:] - target[:, :-1]
            loss_delta_latent = F.mse_loss(d_pred, d_tgt)
        else:
            loss_delta_latent = pred.new_zeros(())

        if self.sigreg_weight > 0:
            reg_source = emb if emb is not None else pred
            if reg_source.ndim != 3:
                raise ValueError(f"SIGReg source must be [B, T, D], got {tuple(reg_source.shape)}")
            loss_sigreg = self.sigreg(reg_source.transpose(0, 1))
        else:
            loss_sigreg = pred.new_zeros(())

        loss = (
            self.mse_weight * loss_mse
            + self.sigreg_weight * loss_sigreg
            + self.delta_latent_weight * loss_delta_latent
        )

        return {
            "loss": loss,
            "loss_mse": loss_mse.detach(),
            "loss_sigreg": loss_sigreg.detach(),
            "loss_delta_latent": loss_delta_latent.detach(),
        }
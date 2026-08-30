from torch import nn

from .backbone import Transformer
from .losses import TacForceLoss
from .tactile_tokenizer import TactileTokenizer
from .wm_condition_encoder import ConditionEncoder


class TacForceWorldModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.chunk_horizon = args.chunk_horizon
        self.future_shift = args.future_shift
        self.model_dim = args.model_dim
        self.tactile_embed_dim = args.tactile_embed_dim
        self.use_delta = args.use_delta_latent
        self.mse_weight = args.mse_weight
        self.sigreg_weight = args.sigreg_weight
        self.delta_latent_weight = getattr(args, "delta_latent_weight", 0.0)

        self.tokenizer = TactileTokenizer(
            in_channels=args.tactile_in_channels,
            hand_dim=args.tactile_hand_dim,
            embed_dim=args.tactile_embed_dim,
            hidden_channels=args.tactile_hidden_channels,
            mid_channels=args.tactile_mid_channels,
            depth=args.tactile_depth,
            heads=args.tactile_heads,
            dropout=args.dropout,
        )
        self.condition_encoder = ConditionEncoder(
            force_dim=args.force_dim,
            state_dim=args.state_dim,
            model_dim=args.cond_dim,
            hidden_dim=args.cond_hidden_dim,
            dropout=args.dropout,
            use_force=args.use_force,
            use_state=args.use_state,
            use_delta_force=args.use_delta_force,
            use_delta_state=args.use_delta_state,
            branch_depth=args.cond_branch_depth,
            kernel_size=args.cond_kernel_size,
        )
        self.backbone = Transformer(
            x_dim=args.tactile_embed_dim,
            cond_dim=args.cond_dim,
            hidden_dim=args.model_dim,
            output_dim=args.model_dim,
            depth=args.depth,
            heads=args.heads,
            dim_head=args.dim_head,
            mlp_dim=args.mlp_dim,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            use_condition=args.use_condition,
        )
        self.pred_head = nn.Sequential(
            nn.LayerNorm(args.model_dim),
            nn.Linear(args.model_dim, args.tactile_embed_dim),
        )
        self.loss_fn = TacForceLoss(
            mse_weight=args.mse_weight,
            sigreg_weight=args.sigreg_weight,
            sigreg_knots=args.sigreg_knots,
            sigreg_num_proj=args.sigreg_num_proj,
            delta_latent_weight=self.delta_latent_weight,
        )

    def encode(self, batch):
        # tactile: [B, T, H, W, C] -> z: [B, T, D]
        tactile_out = self.tokenizer(batch["tactile"].float())
        cond = self.condition_encoder(
            force_4x=batch.get("force_4x"),
            state_4x=batch.get("state_4x"),
            delta_force_4x=batch.get("delta_force_4x"),
            delta_state_4x=batch.get("delta_state_4x"),
        )
        return {
            "z": tactile_out["global"],
            "tokens": tactile_out["tokens"],
            "left_hand": tactile_out["left_hand"],
            "right_hand": tactile_out["right_hand"],
            "cond": cond,
        }

    def predict(self, z, cond):
        h = self.backbone(z, cond)
        pred = self.pred_head(h)
        if self.use_delta:
            pred = z + pred
        return pred

    def _build_chunk_sample(self, z, cond):
        # z, cond: [B, T, D] -> z_in/cond_in/target: [B, H, D]
        _, T, _ = z.shape
        _, tc, _ = cond.shape
        if tc != T:
            raise ValueError("cond must have same time dimension as z")

        H = self.chunk_horizon
        O = self.future_shift
        return z[:, :H], cond[:, :H], z[:, O:O + H]

    def forward(self, batch):
        encoded = self.encode(batch)
        z = encoded["z"]
        cond = encoded["cond"]
        z_in, cond_in, target = self._build_chunk_sample(z, cond)
        pred = self.predict(z_in, cond_in)
        loss_dict = self.loss_fn(pred, target, emb=z)
        return {
            "loss": loss_dict["loss"],
            "loss_mse": loss_dict["loss_mse"],
            "loss_sigreg": loss_dict["loss_sigreg"],
            "loss_delta_latent": loss_dict["loss_delta_latent"],
        }

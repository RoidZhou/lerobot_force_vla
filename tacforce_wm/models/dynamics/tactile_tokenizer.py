import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock2d(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels, eps=1e-6)
        self.norm2 = nn.GroupNorm(groups, channels, eps=1e-6)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return x + h


class Downsample2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class SingleHandSpatialEncoder(nn.Module):
    """
    [N, H, W, C] -> [N, 9, 5, hand_dim]
    """
    def __init__(self, in_channels=3, hand_dim=64, hidden=96, mid=192, dropout=0.0):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.Conv2d(hidden, hidden, 1),
        )

        self.res1 = ResBlock2d(hidden, dropout)
        self.down1 = Downsample2d(hidden)

        self.res2 = ResBlock2d(hidden, dropout)
        self.proj2 = nn.Conv2d(hidden, mid, 3, padding=1)
        self.down2 = Downsample2d(mid)

        self.mid = nn.Sequential(
            ResBlock2d(mid, dropout),
            ResBlock2d(mid, dropout),
            ResBlock2d(mid, dropout),
        )

        self.out_norm = nn.GroupNorm(min(8, mid), mid, eps=1e-6)
        self.conv_out = nn.Conv2d(mid, hand_dim, 3, padding=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        h = self.conv_in(x)
        h = self.res1(h)
        h = self.down1(h)
        h = self.res2(h)
        h = self.proj2(h)
        h = self.down2(h)
        h = self.mid(h)
        h = F.silu(self.out_norm(h))
        h = self.conv_out(h)
        return h.permute(0, 2, 3, 1).contiguous()


class PatchTransformerAggregator(nn.Module):
    def __init__(self, dim=384, depth=6, heads=6, mlp_ratio=4.0, dropout=0.1, num_patches=90, pool="cls"):
        super().__init__()
        assert pool in ["cls", "mean"]
        self.dim = dim
        self.num_patches = num_patches
        self.pool = pool

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, 1 + num_patches, dim) * 0.02)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: [B*T, N, D]
        """
        B, N, D = x.shape
        assert N == self.num_patches

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)              # [B, 1+N, D]
        x = x + self.pos_emb[:, :N + 1]
        x = self.drop(x)

        x = self.encoder(x)
        x = self.norm(x)

        if self.pool == "cls":
            global_token = x[:, 0]
        else:
            global_token = x[:, 1:].mean(dim=1)

        return global_token, x


class TactileTokenizer(nn.Module):
    """
    Input:
      x: [B, T, 36, 20, 6]

    Output:
      x: [B, T, D]
    """
    def __init__(
        self,
        in_channels=3,
        hand_dim=64,
        embed_dim=384,
        hidden_channels=96,
        mid_channels=192,
        depth=6,
        heads=6,
        dropout=0.1,
        pool="cls",
    ):
        super().__init__()

        self.hand_dim = hand_dim
        self.embed_dim = embed_dim

        self.encoder = SingleHandSpatialEncoder(
            in_channels=in_channels,
            hand_dim=hand_dim,
            hidden=hidden_channels,
            mid=mid_channels,
            dropout=dropout,
        )

        # factorized 2D pos + hand identity
        self.row_pos = nn.Parameter(torch.randn(1, 1, 9, 1, hand_dim) * 0.02)
        self.col_pos = nn.Parameter(torch.randn(1, 1, 1, 5, hand_dim) * 0.02)
        self.left_emb = nn.Parameter(torch.randn(1, 1, 1, 1, hand_dim) * 0.02)
        self.right_emb = nn.Parameter(torch.randn(1, 1, 1, 1, hand_dim) * 0.02)

        self.token_proj = nn.Sequential(
            nn.LayerNorm(hand_dim),
            nn.Linear(hand_dim, embed_dim),
        )

        self.transformer = PatchTransformerAggregator(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            dropout=dropout,
            num_patches=90,
            pool=pool,
        )

    def encode_hands(self, x):
        B, T, H, W, C = x.shape
        x = x.reshape(B * T, H, W, C)

        left = x[..., :3]
        right = x[..., 3:]

        left = self.encoder(left)
        right = self.encoder(right)

        left = left.reshape(B, T, 9, 5, self.hand_dim)
        right = right.reshape(B, T, 9, 5, self.hand_dim)
        return left, right

    def tokenize(self, left, right):
        B, T, H, W, D = left.shape

        left = left + self.row_pos + self.col_pos + self.left_emb
        right = right + self.row_pos + self.col_pos + self.right_emb

        left = left.reshape(B * T, H * W, D)   # [B*T,45,D]
        right = right.reshape(B * T, H * W, D)

        tokens = torch.cat([left, right], dim=1)   # [B*T,90,D]
        patch_tokens = self.token_proj(tokens)      # [B*T,90,embed_dim]

        return patch_tokens, left, right

    def forward(self, x):
        B, T, _, _, _ = x.shape

        left, right = self.encode_hands(x)
        patch_tokens_bt, _, _ = self.tokenize(left, right)

        global_bt, all_tokens_bt = self.transformer(patch_tokens_bt)

        global_token = global_bt.reshape(B, T, self.embed_dim)
        all_tokens = all_tokens_bt.reshape(B, T, 91, self.embed_dim)

        return {
            "global": global_token,
            "tokens": all_tokens,
            "left_hand": left,
            "right_hand": right,
        }
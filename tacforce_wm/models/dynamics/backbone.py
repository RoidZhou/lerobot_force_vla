import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = dropout

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, causal=True):
        B, T, _ = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv]

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal
        )

        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )

        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(c).chunk(6, dim=-1)
        # Attention
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        # MLP
        x = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        x_dim,
        cond_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        max_seq_len=64,
        dropout=0.0,
        use_condition=False,
    ):
        super().__init__()

        self.use_condition = use_condition

        self.input_proj = (
            nn.Linear(x_dim, hidden_dim) if x_dim != hidden_dim else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        )
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        if self.use_condition:
            self.cond_proj = (
                nn.Linear(cond_dim, hidden_dim) if cond_dim != hidden_dim else nn.Identity()
            )
            block_cls = ConditionalBlock
        else:
            self.cond_proj = None
            block_cls = Block

        self.layers = nn.ModuleList([
            block_cls(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, c=None):
        B, T, _ = x.shape

        x = self.input_proj(x)
        x = x + self.pos_emb[:, :T]

        if self.use_condition:
            assert c is not None, "Condition required but not provided"
            c = self.cond_proj(c)

        for block in self.layers:
            if self.use_condition:
                x = block(x, c)
            else:
                x = block(x)

        x = self.norm(x)
        x = self.output_proj(x)
        return x
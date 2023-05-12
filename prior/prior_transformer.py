import math
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from rotary_embedding_torch import RotaryEmbedding

from torch import nn


class LayerNorm(nn.LayerNorm):
    """
    Implementation that supports fp16 inputs but fp32 gains/biases.
    """

    def forward(self, x: torch.Tensor):
        return super().forward(x.float()).to(x.dtype)


class MLP(pl.LightningModule):
    """
    Simple MLP.

    Args:
        dim_in (int): input dimension
        dim_out (int): output dimension
        dim_hidden (int): hidden dimension
        hidden_depth (int): number of hidden layers
    """

    def __init__(
        self, dim_in: int, dim_out: int, expansion_factor: int, hidden_depth: int
    ):
        super(MLP, self).__init__()

        dim_hidden = dim_in * expansion_factor

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.hidden_depth = hidden_depth

        # first layer to get into hidden space
        self.layers = [nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.SiLU())]

        # hidden layers
        for _ in range(hidden_depth):
            self.layers.append(
                nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.SiLU())
            )

        # final layer to get into output space
        self.layers.append(nn.Linear(dim_hidden, dim_out))
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)


class SwiGLU(nn.Module):
    """used successfully in https://arxiv.org/abs/2204.0231"""

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


def FeedForward(dim, mult=4, dropout=0.0, post_activation_norm=False):
    """post-activation norm https://arxiv.org/abs/2110.09456"""

    inner_dim = int(mult * dim)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        SwiGLU(),
        LayerNorm(inner_dim) if post_activation_norm else nn.Identity(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


class SelfAttention(pl.LightningModule):
    """
    Causal self-attention layer.

    Args:
        num_heads (int): number of attention heads
        emb_dim (int): input dimension
        bias (bool): whether to use bias
        is_causal (bool): whether to use causal masking
        dropout (float): dropout probability
    """

    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        causal: bool,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super(SelfAttention, self).__init__()
        assert (
            emb_dim % num_heads == 0
        ), f"emb_dim must be divisible by num_heads: got {emb_dim} % {num_heads} which is {emb_dim % num_heads}"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(emb_dim, 3 * emb_dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        # regularization
        self.causal = causal
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.rotary_emb = RotaryEmbedding(dim=32)

    def forward(self, x):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        dropout = self.dropout if self.training else 0.0

        query, key = map(self.rotary_emb.rotate_queries_or_keys, (query, key))

        y = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=dropout,
            is_causal=self.causal,
        )
        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))

        return y


class ResidualAttentionBlock(pl.LightningModule):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        causal: bool,
        bias: bool = False,
        dropout: float = 0.0,
        mlp_expansion_factor: int = 4,
        norm_in: bool = True,
        norm_out: bool = True,
    ):
        super(ResidualAttentionBlock, self).__init__()
        self.attn = SelfAttention(
            num_heads,
            emb_dim,
            bias=bias,
            dropout=dropout,
            causal=causal,
        )
        self.ln_1 = LayerNorm(emb_dim) if norm_in else nn.Identity()
        self.ln_2 = LayerNorm(emb_dim) if norm_out else nn.Identity()
        self.ff = FeedForward(
            emb_dim,
            mult=mlp_expansion_factor,
            dropout=dropout,
            post_activation_norm=True,
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x


class Transformer(pl.LightningModule):
    def __init__(
        self,
        emb_dim: int,
        num_layers: int,
        num_heads: int,
        causal: bool,
        bias: bool = False,
        dropout: float = 0.0,
        mlp_expansion_factor: int = 4,
    ):
        super(Transformer, self).__init__()
        self.emb_dim = emb_dim
        self.layers = num_layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    bias=bias,
                    dropout=dropout,
                    mlp_expansion_factor=mlp_expansion_factor,
                    causal=causal,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        return x


class PriorTransformer(pl.LightningModule):
    """
    A Causal Transformer that conditions on CLIP text embedding, text.

    :param ctx_len: number of text tokens to expect.
    :param emb_dim: width of the transformer.
    :param num_layers: depth of the transformer.
    :param num_heads: heads in the transformer.
    :param final_ln: use a LayerNorm after the output layer.
    :param clip_dim: dimension of clip feature.
    """

    def __init__(
        self,
        ctx_len,
        emb_dim,
        num_layers,
        num_heads,
        final_ln,
        clip_dim,
        bias=False,
        dropout=0.0,
        ml_expansion_factor=4,
        num_diffusion_timesteps=1000,
        causal=True,
    ):
        super(PriorTransformer, self).__init__()

        self.ctx_len = ctx_len
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.clip_dim = clip_dim
        self.ext_len = 4

        self.dim = emb_dim
        self.self_cond = False

        self.to_time_embed = nn.Embedding(num_diffusion_timesteps, emb_dim)

        needs_projection = clip_dim != emb_dim

        self.text_enc_proj = (
            nn.Linear(clip_dim, emb_dim) if needs_projection else nn.Identity()
        )
        self.text_emb_proj = (
            nn.Linear(clip_dim, emb_dim) if needs_projection else nn.Identity()
        )
        self.clip_img_proj = (
            nn.Linear(clip_dim, emb_dim) if needs_projection else nn.Identity()
        )

        self.out_proj = nn.Linear(emb_dim, clip_dim)

        self.transformer = Transformer(
            emb_dim=emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            bias=bias,
            dropout=dropout,
            mlp_expansion_factor=ml_expansion_factor,
            causal=causal,
        )

        self.final_ln = LayerNorm(emb_dim) if final_ln else nn.Identity()

        self.prd_emb = nn.Parameter(torch.randn((1, 1, emb_dim)))

    def forward(
        self,
        x,
        timesteps,
        text_emb=None,
        text_enc=None,
    ):
        bsz = x.shape[0]

        time_emb = self.to_time_embed(timesteps)
        text_enc = self.text_enc_proj(text_enc)
        text_emb = self.text_emb_proj(text_emb)
        x = self.clip_img_proj(x)

        input_seq = [
            text_enc,
            text_emb.unsqueeze(1),
            time_emb.unsqueeze(1),
            x.unsqueeze(1),
            self.prd_emb.to(x.dtype).expand(bsz, -1, -1),
        ]

        input = torch.cat(input_seq, dim=1)

        out = self.transformer(input)

        out = self.final_ln(out)

        # pull the last "token" which is our prediction
        out = self.out_proj(out[..., -1, :])

        return out

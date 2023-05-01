import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from torch import nn


def timestep_embedding(steps, emb_dim, max_period=10000):
    """
    Creates a sinusoidal embedding of the given `steps`.

    Args:
        steps (Tensor): A 1-D Tensor of `N` indices, one per batch element. These may be fractional.
        emb_dim (int): The dimension of the output embeddings.
        max_period (float, optional): The maximum period of the sinusoidal func. Controls the minimum freq. of the embeddings.

    Returns:
        Tensor: A [N x emb_dim] Tensor of sinusoidal embeddings, where each element is a function of the corresponding `steps`.
    """
    device = steps.device
    half = emb_dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, device=device, dtype=torch.float32))
        * torch.arange(start=0, end=half, device=device, dtype=torch.float32)
        / half
    )
    args = steps.unsqueeze(-1).float() * freqs.unsqueeze(0)
    embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
    if emb_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


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


class CausalSelfAttention(pl.LightningModule):
    """
    Causal self-attention layer.

    (from pytorch documentation)

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
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super(CausalSelfAttention, self).__init__()
        assert (
            emb_dim % num_heads == 0
        ), f"emb_dim must be divisible by num_heads: got {emb_dim} % {num_heads} which is {emb_dim % num_heads}"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(emb_dim, 3 * emb_dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        # regularization
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.emb_dim = emb_dim

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

        y = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=dropout,
            is_causal=True,
        )
        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))

        return y


class ResidualAttentionBlock(pl.LightningModule):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        bias: bool = False,
        dropout: float = 0.0,
        mlp_expansion_factor: int = 4,
        mlp_hidden_depth: int = 1,
    ):
        super(ResidualAttentionBlock, self).__init__()
        self.attn = CausalSelfAttention(
            num_heads,
            emb_dim,
            bias=bias,
            dropout=dropout,
        )
        self.ln_1 = LayerNorm(emb_dim)
        self.ln_2 = LayerNorm(emb_dim)
        self.mlp = MLP(
            dim_in=emb_dim,
            dim_out=emb_dim,
            expansion_factor=mlp_expansion_factor,
            hidden_depth=mlp_hidden_depth,
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(pl.LightningModule):
    def __init__(
        self,
        emb_dim: int,
        num_layers: int,
        num_heads: int,
        bias: bool = False,
        dropout: float = 0.0,
        mlp_expansion_factor: int = 4,
        mlp_hidden_depth: int = 1,
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
                    mlp_hidden_depth=mlp_hidden_depth,
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
        mlp_hidden_depth=1,
    ):
        super(PriorTransformer, self).__init__()

        self.ctx_len = ctx_len
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.clip_dim = clip_dim
        self.ext_len = 4

        self.time_embed = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.text_enc_proj = nn.Linear(clip_dim, emb_dim)
        self.text_emb_proj = nn.Linear(clip_dim, emb_dim)
        self.clip_img_proj = nn.Linear(clip_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, clip_dim)
        self.transformer = Transformer(
            emb_dim=emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            bias=bias,
            dropout=dropout,
            mlp_expansion_factor=ml_expansion_factor,
            mlp_hidden_depth=mlp_hidden_depth,
        )
        if final_ln:
            self.final_ln = LayerNorm(emb_dim)
        else:
            self.final_ln = None

        self.positional_embedding = nn.Parameter(
            torch.empty(1, ctx_len + self.ext_len, emb_dim)
        )
        self.prd_emb = nn.Parameter(torch.randn((1, 1, emb_dim)))

        nn.init.normal_(self.prd_emb, std=0.01)
        nn.init.normal_(self.positional_embedding, std=0.01)

    def forward(
        self,
        x,
        timesteps,
        text_emb=None,
        text_enc=None,
    ):
        bsz = x.shape[0]

        time_emb = self.time_embed(timestep_embedding(timesteps, self.emb_dim))
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
        pos_emb = self.positional_embedding.to(input.dtype)
        input = input + pos_emb  # (B, ctx_len+4, emb_dim)
        out = self.transformer(input)

        if self.final_ln is not None:
            out = self.final_ln(out)

        out = self.out_proj(out[:, -1])

        return out

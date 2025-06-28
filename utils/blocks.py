import torch
from torch import nn
from torch import Tensor

from torch import nn, einsum

from einops import rearrange

from zeta.utils import exists

from zeta.nn import (
    MambaBlock,
    FeedForward,
    MultiQueryAttention,
)
import torch.nn.functional as F


class LinearAttention(nn.Module):
    def __init__(self, dim, *, heads=4, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h),
            (q, k, v),
        )

        q = q * self.scale
        q, k = q.softmax(dim=-1), k.softmax(dim=-2)

        if exists(mask):
            k.masked_fill_(mask, 0.0)

        context = einsum("b n d, b n e -> b d e", q, k)
        out = einsum("b d e, b n d -> b n e", context, v)
        out = rearrange(out, " (b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** (-0.5)
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.g


class TransformerBlock(nn.Module):
    """
    TransformerBlock is a module that represents a single block of the Multi-Query Transformer.
    It consists of a multi-query attention layer, a feed-forward network, and layer normalization.

    Args:
        dim (int): The input and output dimension of the block.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        ff_mult (int, optional): The multiplier for the feed-forward network dimension. Defaults to 4.

    Attributes:
        dim (int): The input and output dimension of the block.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        dropout (float): The dropout probability.
        ff_mult (int): The multiplier for the feed-forward network dimension.
        attn (MultiQueryAttention): The multi-query attention layer.
        ffn (FeedForward): The feed-forward network.
        norm (nn.LayerNorm): The layer normalization.

    Methods:
        forward(x: Tensor) -> Tensor:
            Performs a forward pass of the TransformerBlock.

    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        use_linear_attn: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.use_linear_attn = use_linear_attn

        self.attn = MultiQueryAttention(dim, heads, *args, **kwargs)

        # Linear Attention
        self.linear_attn = LinearAttention(
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout
        )

        self.ffn = FeedForward(dim, dim, ff_mult, *args, **kwargs)

        # Normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass of the TransformerBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        if self.use_linear_attn:
            
            x = self.linear_attn(x)
            
            x = self.norm(x)
            x = self.ffn(x)
        else:
            x, _, _ = self.attn(x)
            x = self.norm(x)
            x = self.ffn(x)

        return x


class MambaTransformerblock(nn.Module):
    """
    MambaTransformerblock is a module that represents a block in the Mamba Transformer model.

    Args:
        dim (int): The input dimension of the block.
        heads (int): The number of attention heads in the block.
        depth (int): The number of layers in the block.
        dim_head (int): The dimension of each attention head.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        ff_mult (int, optional): The multiplier for the feed-forward network dimension. Defaults to 4.
        d_state (int, optional): The dimension of the state. Defaults to None.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension of the block.
        depth (int): The number of layers in the block.
        dim_head (int): The dimension of each attention head.
        d_state (int): The dimension of the state.
        dropout (float): The dropout rate.
        ff_mult (int): The multiplier for the feed-forward network dimension.
        mamba_blocks (nn.ModuleList): List of MambaBlock instances.
        transformer_blocks (nn.ModuleList): List of TransformerBlock instances.
        ffn_blocks (nn.ModuleList): List of FeedForward instances.
        norm (nn.LayerNorm): Layer normalization module.

    Examples:
        import torch
        from mt import MambaTransformerblock

        x = torch.randn(1, 10, 512)
        model = MambaTransformerblock(
            dim=512,
            heads=8,
            depth=4,
            dim_head=64,
            d_state=512,
            dropout=0.1,
            ff_mult=4
        )
        print(model(x).shape)


    """

    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int,
        dim_head: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        d_state: int = None,
        transformer_depth: int = 1,
        mamba_depth: int = 1,
        use_linear_attn: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.dim_head = dim_head
        self.d_state = d_state
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.d_state = d_state
        self.transformer_depth = transformer_depth
        self.mamba_depth = mamba_depth

        # Mamba, Transformer, and ffn blocks
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(dim, mamba_depth, d_state, *args, **kwargs)
            for _ in range(mamba_depth)
        ])
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim,
                heads,
                dim_head,
                dropout,
                ff_mult,
                use_linear_attn,
                *args,
                **kwargs,
            ) for _ in range(transformer_depth)
        ])

        self.ffn_blocks = nn.ModuleList([
            FeedForward(dim, dim, ff_mult, *args, **kwargs)
            for _ in range(depth)
        ])

        # Layernorm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        for mamba, attn, ffn in zip(
            self.mamba_blocks,
            self.transformer_blocks,
            self.ffn_blocks,
        ):
            x = self.norm(x)
            x = mamba(x) + x
            x = self.norm(x)
            x = attn(x) + x
            x = self.norm(x)
            x = ffn(x) + x

        return x

class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.projection = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: Tensor) -> Tensor:
        # Project the 3D volume into patches and flatten spatial dimensions
        x = self.projection(x)  # [B, embed_dim, D, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class ConvEmbedding3D(nn.Module):
    """
    3D Convolutional Embedding to project the input volume to a desired embedding dimension 
    while maintaining the same spatial dimensions.
    """
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.projection = nn.Conv3d(
            in_channels, embed_dim, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        # Project the input volume
        return self.projection(x)  # [B, embed_dim, D, H, W]





import torch
from torch import nn
from torch import Tensor

from torch import nn

from utils.blocks import *

class MambaTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        heads: int,
        depth: int,
        dim_head: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        d_state: int = None,
        return_embeddings: bool = False,
        transformer_depth: int = 1,
        mamba_depth: int = 1,
        use_linear_attn=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = embed_dim
        self.depth = depth
        self.dim_head = dim_head
        self.d_state = d_state
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.return_embeddings = return_embeddings
        self.transformer_depth = transformer_depth
        self.mamba_depth = mamba_depth

        # Use a convolutional embedding to maintain spatial dimensions
        self.emb = ConvEmbedding3D(in_channels, embed_dim)
        
        # Transformer block
        self.mt_block = MambaTransformerblock(
            embed_dim,
            heads,
            depth,
            dim_head,
            dropout,
            ff_mult,
            d_state,
            return_embeddings,
            transformer_depth,
            mamba_depth,
            use_linear_attn,
            *args,
            **kwargs,
        )

        # Project back to the original input channel size
        self.to_output = nn.Conv3d(
            embed_dim, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MambaTransformer model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            Tensor: Output tensor of the same shape as the input.
        """

    
        # Embed the input volume
        x = self.emb(x)
        print(x.shape)
        # Apply the transformer block
        x = self.mt_block(x)

        # Project back to the original channel dimension
        x = self.to_output(x)

        return x

# Input tensor of shape [B, C, D, H, W]
input_volume = torch.randn(1, 1, 16, 32, 32)  # Example 3D volume

# Model instantiation
model = MambaTransformer(
    in_channels=1,         # Number of input channels
    embed_dim=64,          # Embedding dimension
    heads=8,               # Number of attention heads
    depth=4,               # Number of transformer layers
    dim_head=64,           # Dimension of each attention head
    d_state=512,           # Dimension of the state
    dropout=0.1,           # Dropout rate
    ff_mult=4,             # Multiplier for feed-forward layer dimension
    return_embeddings=False,
    transformer_depth=2,
    mamba_depth=10,
    use_linear_attn=True,
)

# Forward pass
output_volume = model(input_volume)



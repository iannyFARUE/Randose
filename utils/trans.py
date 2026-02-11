import torch
import torch.nn as nn

from utils.blocks import *

from zeta.nn import (
    MambaBlock,
    FeedForward,
    MultiQueryAttention,
)


class TransformerBlock3D(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout, ff_mult, mamba_depth=1, d_state=512, *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.mamba_depth = mamba_depth
        self.d_state = d_state

        # Mamba block
        
        
        # Attention mechanism
        self.attn = MultiQueryAttention(dim, heads, *args, **kwargs)
        
        # Feed-forward network
        self.ffn = FeedForward(dim, dim, ff_mult, *args, **kwargs)
        
        # Normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        # Apply MambaBlock before attention
        x = self.mamba(x) + x

        # Apply attention layer
        x, _, _ = self.attn(x)

        # Apply normalization
        x = self.norm(x)

        # Apply feed-forward network
        x = self.ffn(x) + x

        return x


# Testing with a random 3D volume input
batch_size = 2
depth = 128
height = 128
width = 128
channels = 16

# Create a random tensor representing the 3D volume (batch_size, depth, height, width, channels)
x = torch.randn(batch_size, depth, height, width, channels)

# Initialize the TransformerBlock3D
transformer_block = TransformerBlock3D(
    dim=channels, heads=4, dim_head=4, dropout=0.1, ff_mult=4,  d_state=512
)

# Forward pass through the transformer block
output = transformer_block(x)

# Check the output shape
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
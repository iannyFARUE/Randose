import torch
from torch import nn, Tensor


from blocks import *


import numpy as np

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


class MambaFormer(nn.Module):
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

        # Stack multiple MambaTransformerblocks
        self.mt_blocks = nn.ModuleList([
            MambaTransformerblock(
                embed_dim,
                heads,
                transformer_depth,  # Number of transformer layers in each block
                dim_head,
                dropout,
                ff_mult,
                d_state,
                return_embeddings,
                *args,
                **kwargs
            ) for _ in range(mamba_depth)  # Stack `mamba_depth` transformer blocks
        ])

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

        print("Input", x.shape)
        # Embed the input volume (project to higher-dimensional space)
        x = self.emb(x)

        print("Embed", x.shape)
        # Pass the input through multiple transformer blocks
        for block in self.mt_blocks:
            x = block(x)

        print("Ma", x.shape)
        # Project the output back to the original channel size (for dose prediction)
        x = self.to_output(x)
        print("out", x.shape)
        return x
    
# Input tensor of shape [B, C, D, H, W]
input_volume = torch.randn(1, 1, 16, 32, 32)  # Example 3D volume

class Model_Mambaformer(nn.Module):
    def __init__(self, in_ch, out_ch, list_ch_A, list_ch_B, d_state, d_conv, expand, channel_token=False):
        super(Model_Mambaformer, self).__init__()
        self.net_A = MambaFormer(
                    in_channels=in_ch,         # Number of input channels
                    embed_dim=64,          # Embedding dimension
                    heads=8,               # Number of attention heads
                    depth=4,               # Number of transformer layers
                    dim_head=64,           # Dimension of each attention head
                    d_state=512,           # Dimension of the state
                    dropout=0.1,           # Dropout rate
                    ff_mult=4,             # Multiplier for feed-forward layer dimension
                    return_embeddings=False,
                    transformer_depth=8,
                    mamba_depth=10,
                    use_linear_attn=True,
                )
        
        self.conv_out_A = nn.Conv3d(list_ch_A[1], out_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out_net_A = self.net_A(x)
        
        output_A = self.conv_out_A(out_net_A)
        return output_A

if __name__ == '__main__':
    model = Model_Mambaformer(in_ch=9, out_ch=1,
                              list_ch_A=[-1, 16, 32, 64, 128, 256],
                              list_ch_B=[-1, 32, 64, 128, 256, 512],
                              d_state=16, d_conv=4, expand=2, channel_token=False)

    input_tensor = torch.randn(1, 9, 16, 32, 32)  # Adjust input channels to match `in_ch=9`
    
    output = model(input_tensor)  # Run forward pass

    a = np.sum(np.prod(v.size()) for v in model.parameters()) * 4e-6
    print("Model output shape:", output.shape)
    print("Total parameter size (MB):", a)



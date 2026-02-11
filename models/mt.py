import torch
import torch.nn as nn
#from mamba_ssm import Mamba
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.blocks import *

from utils.mambat import *

class ResidualMambaBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_sizes, stride, norm_op, nonlin, 
                 d_state, d_conv, expand, channel_token=False, embed_dim=64, heads=8, depth=1):
        super().__init__()

        # Parallel convolutional layers
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_sizes[0], 
                               stride=stride, padding=kernel_sizes[0] // 2)
        self.conv2 = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_sizes[1], 
                               stride=stride, padding=kernel_sizes[1] // 2)
        self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_sizes[2], 
                               stride=stride, padding=kernel_sizes[2] // 2)

        self.mamba_transformer = MambaTransformer(
                                in_channels=output_channels,         # Number of input channels
                                embed_dim=embed_dim,          # Embedding dimension
                                heads=heads,               # Number of attention heads
                                depth=depth,               # Number of transformer layers
                                dim_head=64,           # Dimension of each attention head
                                d_state=d_state,           # Dimension of the state
                                dropout=0.1,           # Dropout rate
                                ff_mult=4,             # Multiplier for feed-forward layer dimension
                                return_embeddings=False,
                                transformer_depth=2,
                                mamba_depth=1,
                                use_linear_attn=True,
                                )
        # MambaTransformer as a parallel path
      

        # Normalization and Non-linearity layers
        self.norm1 = norm_op(16)
        self.norm2 = norm_op(16)
        self.norm3 = norm_op(16)
        self.norm_mamba = norm_op(output_channels)
        self.nonlin = nonlin()

        # Shortcut to handle channel mismatch
        self.shortcut = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride) \
            if input_channels != output_channels or stride != 1 else None

        self.channel_token = channel_token

    def forward(self, x):
        # Parallel convolutional layers
        out1 = self.nonlin(self.conv1(x))
        out2 = self.nonlin(self.conv2(x))
        out3 = self.nonlin(self.conv3(x))

        # MambaTransformer
        mamba_out1 = self.mamba_transformer(out1)
        mamba_out2 = self.mamba_transformer(out2)
        mamba_out3 = self.mamba_transformer(out3)
        

        # Merge all outputs
        merged = mamba_out1 + mamba_out2 + mamba_out3

        # Add shortcut connection
        if self.shortcut:
            shortcut = self.shortcut(x)
            merged += shortcut

        return merged

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=True),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_ch, list_ch, d_state, d_conv, expand, channel_token=False):
        super(Encoder, self).__init__()

        # Replace the existing SingleConv with ResidualMambaBlock
        self.encoder_1 = ResidualMambaBlock(in_ch, list_ch[1], kernel_sizes=[3, 5, 7], stride=1,
                                             norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU,
                                             d_state=d_state, d_conv=d_conv, expand=expand, channel_token=channel_token)
        
        self.encoder_2 = ResidualMambaBlock(list_ch[1], list_ch[2], kernel_sizes=[3, 5, 7], stride=2,
                                             norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU,
                                             d_state=d_state, d_conv=d_conv, expand=expand, channel_token=channel_token)
        
        self.encoder_3 = ResidualMambaBlock(list_ch[2], list_ch[3], kernel_sizes=[3, 5, 7], stride=2,
                                             norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU,
                                             d_state=d_state, d_conv=d_conv, expand=expand, channel_token=channel_token)
        
        self.encoder_4 = ResidualMambaBlock(list_ch[3], list_ch[4], kernel_sizes=[3, 5, 7], stride=2,
                                             norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU,
                                             d_state=d_state, d_conv=d_conv, expand=expand, channel_token=channel_token)
        
        self.encoder_5 = ResidualMambaBlock(list_ch[4], list_ch[5], kernel_sizes=[3, 5, 7], stride=2,
                                             norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU,
                                             d_state=d_state, d_conv=d_conv, expand=expand, channel_token=channel_token)

    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)
        out_encoder_5 = self.encoder_5(out_encoder_4)


        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5]

class AddFeatureMaps(nn.Module):
    def __init__(self):
        """
        A simple class to combine two feature maps by element-wise addition.
        """
        super(AddFeatureMaps, self).__init__()

    def forward(self, x1, x2):
        """
        Forward pass to combine two feature maps by element-wise addition.

        Args:
            x1 (Tensor): The first feature map tensor.
            x2 (Tensor): The second feature map tensor.

        Returns:
            Tensor: The combined feature map.
        """
        # Ensure the feature maps have the same shape for element-wise addition
        if x1.shape != x2.shape:
            raise ValueError("Feature maps must have the same shape for addition")
        
        combined = x1 + x2
        return combined
    
class Decoder(nn.Module):
    def __init__(self, list_ch, d_state, d_conv, expand, channel_token=False):
        super(Decoder, self).__init__()

        self.upconv_4 = UpConv(list_ch[5], list_ch[4])

        self.decoder_conv_4 = ResidualMambaBlock(list_ch[4], list_ch[4], kernel_sizes=[3, 5, 7], stride=1, norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU, d_state=d_state, d_conv=d_conv, expand=expand, channel_token=channel_token)
       
        self.upconv_3 = UpConv(list_ch[4], list_ch[3])

        self.decoder_conv_3 = ResidualMambaBlock(list_ch[3], list_ch[3], kernel_sizes=[3, 5, 7], stride=1, norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU, d_state=d_state, d_conv=d_conv, expand=expand, channel_token=channel_token)
        
        self.upconv_2 = UpConv(list_ch[3], list_ch[2])
        
        self.decoder_conv_2 = ResidualMambaBlock(list_ch[2], list_ch[2], kernel_sizes=[3, 5, 7], stride=1, norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU, d_state=d_state, d_conv=d_conv, expand=expand, channel_token=channel_token)
        
        self.upconv_1 = UpConv(list_ch[2], list_ch[1])

        self.decoder_conv_1 = ResidualMambaBlock(list_ch[1], list_ch[1], kernel_sizes=[3, 5, 7], stride=1, norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU, d_state=d_state, d_conv=d_conv, expand=expand, channel_token=channel_token)
        
        self.combiner = AddFeatureMaps()
      
    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5 = out_encoder
        
        out_decoder_4 = self.decoder_conv_4(self.combiner(self.upconv_4(out_encoder_5), out_encoder_4))

        out_decoder_3 = self.decoder_conv_3(self.combiner(self.upconv_3(out_decoder_4), out_encoder_3))
        
        out_decoder_2 = self.decoder_conv_2(self.combiner(self.upconv_2(out_decoder_3), out_encoder_2))
        
        out_decoder_1 = self.decoder_conv_1(self.combiner(self.upconv_1(out_decoder_2), out_encoder_1))
        
    
        

        return out_decoder_1

class BaseUNet(nn.Module):
    def __init__(self, in_ch, list_ch, d_state, d_conv, expand, channel_token=False):
        super(BaseUNet, self).__init__()
        self.encoder = Encoder(in_ch, list_ch, d_state, d_conv, expand, channel_token)
        self.decoder = Decoder(list_ch, d_state, d_conv, expand, channel_token)

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)
        return out_decoder

class Model_MT(nn.Module):
    def __init__(self, in_ch, out_ch, list_ch_A, list_ch_B, d_state, d_conv, expand, channel_token=False):
        super(Model_MT, self).__init__()
        self.net_A = BaseUNet(in_ch, list_ch_A, d_state, d_conv, expand, channel_token)
        self.conv_out_A = nn.Conv3d(list_ch_A[1], out_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out_net_A = self.net_A(x)
        
        output_A = self.conv_out_A(out_net_A)
        return output_A

if __name__ == '__main__':
    model = Model_MT(in_ch=9, out_ch=1,
                  list_ch_A=[-1, 16, 32, 64, 128, 256],
                  list_ch_B=[-1, 32, 64, 128, 256, 512],
                  d_state=16, d_conv=4, expand=2, channel_token=False)
    a = np.sum(np.prod(v.size()) for v in model.parameters())*4e-6
    print(a)
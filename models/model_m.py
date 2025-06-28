import torch
import torch.nn as nn
#from mamba_ssm import Mamba
import numpy as np

class ResidualMambaBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_sizes, stride, norm_op, nonlin, 
                 d_state, d_conv, expand, channel_token=False):
        super().__init__()

        # Define the parallel convolutional layers with adjusted padding
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_sizes[0], 
                               stride=stride, padding=kernel_sizes[0] // 2)
        self.conv2 = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_sizes[1], 
                               stride=stride, padding=kernel_sizes[1] // 2)
        self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_sizes[2], 
                               stride=stride, padding=kernel_sizes[2] // 2)

        # Normalization and Non-linearity layers
        self.norm1 = norm_op(output_channels)
        self.norm2 = norm_op(output_channels)
        self.norm3 = norm_op(output_channels)
        self.nonlin = nonlin()

        # Shortcut to handle channel mismatch
        self.shortcut = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride) \
            if input_channels != output_channels or stride != 1 else None

        
        

        # Mamba Layer
        #self.mamba_layer = Mamba(
        #    d_model=output_channels,  # Model dimension
        #    d_state=d_state,          # SSM state expansion factor
        #    d_conv=d_conv,            # Local convolution width
        #    expand=expand             # Block expansion factor
        #)
        self.channel_token = channel_token
        
    def forward(self, x):
        # Apply the three parallel conv layers
        out1 = self.nonlin(self.norm1(self.conv1(x)))
     

        out2 = self.nonlin(self.norm2(self.conv2(x)))


        out3 = self.nonlin(self.norm3(self.conv3(x)))



        # Merge the outputs (e.g., by addition)
        merged = out1 + out2 + out3  # Outputs have matching dimensions due to adjusted padding

        # Apply shortcut (if needed) and add residual
#        identity = self.shortcut(x) #if self.shortcut else x
        out = merged# + identity

        # Mamba Layer
        #B, C, *dims = out.shape  # B: batch size, C: channels, dims: spatial dimensions
        #if self.channel_token:
            # Treat channel as token
        #    out = out.flatten(2).transpose(1, 2)  # Reshape to (B, Tokens, Features)
        #    out = self.mamba_layer(out)
        #    out = out.transpose(1, 2).view(B, C, *dims)
        #else:
        #    # Treat spatial as token
        #    out = out.view(B, C, -1).transpose(1, 2)  # Reshape to (B, Tokens, Features)
        #    out = self.mamba_layer(out)
        #    out = out.transpose(1, 2).view(B, C, *dims)
        
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Model_M(nn.Module):
    def __init__(self, in_ch, out_ch, list_ch_A, list_ch_B, d_state, d_conv, expand, channel_token=False):
        super(Model_M, self).__init__()
        self.net_A = BaseUNet(in_ch, list_ch_A, d_state, d_conv, expand, channel_token)
        self.conv_out_A = nn.Conv3d(list_ch_A[1], out_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out_net_A = self.net_A(x)
        
        output_A = self.conv_out_A(out_net_A)
        return output_A

if __name__ == '__main__':
    model = Model_M(in_ch=9, out_ch=1,
                  list_ch_A=[-1, 16, 32, 64, 128, 256],
                  list_ch_B=[-1, 32, 64, 128, 256, 512],
                  d_state=16, d_conv=4, expand=2, channel_token=False)
    a = np.sum(np.prod(v.size()) for v in model.parameters())*4e-6
    print(a)
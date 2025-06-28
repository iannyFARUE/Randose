import torch
import torch.nn as nn
#from mamba_ssm import Mamba
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from se3d import *

import sys
class ResidualMambaBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_sizes, stride, norm_op, nonlin, 
                 d_state, d_conv, expand, channel_token=False, embed_dim=64, heads=8, depth=1, se_type='CSE3D'):
        super(ResidualMambaBlock, self).__init__()

        # Parallel convolutional layers
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_sizes[0], 
                               stride=stride, padding=kernel_sizes[0] // 2)
        self.conv2 = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_sizes[1], 
                               stride=stride, padding=kernel_sizes[1] // 2)
        self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_sizes[2], 
                               stride=stride, padding=kernel_sizes[2] // 2)
        self.conv4 = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_sizes[3], 
                               stride=stride, padding=kernel_sizes[3] // 2)

        # Normalization and Non-linearity layers
        self.norm1 = norm_op(16)
        self.norm2 = norm_op(16)
        self.norm3 = norm_op(16)
        self.norm_mamba = norm_op(output_channels)
        self.nonlin = nonlin()

        # Shortcut to handle channel mismatch
        self.shortcut = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride) \
            if input_channels != output_channels or stride != 1 else None

        # SE block selection
        if se_type == 'CSE3D':
            self.se_block = ChannelSELayer3D(output_channels, reduction_ratio=2)
        elif se_type == 'SSE3D':
            self.se_block = SpatialSELayer3D(output_channels)
        elif se_type == 'CSSE3D':
            self.se_block = ChannelSpatialSELayer3D(output_channels, reduction_ratio=2)
        elif se_type == 'PE':
            self.se_block = ProjectExciteLayer(output_channels, reduction_ratio=2)
        else:
            self.se_block = None  # No SE block


    def forward(self, x):
        # Parallel convolutional layers
        out1 = self.nonlin(self.conv1(x))
        out2 = self.nonlin(self.conv2(x))
        out3 = self.nonlin(self.conv3(x))
        out4 = self.nonlin(self.conv4(x))

        out = out1 + out2 + out3 + out4

        print("After Multiscale:", out.shape)

        # Apply Squeeze-and-Excitation block (if specified)
        if self.se_block:
            out = self.se_block(out)

        # Add shortcut connection
        if self.shortcut:
            shortcut = self.shortcut(x)
            out += shortcut

        return out



class CrossAttentionFusionModule(nn.Module):
    def __init__(self, in_channels_ct=1, in_channels_ptv=1, in_channels_oar=7, hidden_dim=64, out_channels=1):
        super(CrossAttentionFusionModule, self).__init__()

        # Feature extractors (3D convolutions to process the data from each modality)
        self.ct_conv = nn.Conv3d(in_channels_ct, hidden_dim, kernel_size=3, padding=1)
        self.ptv_conv = nn.Conv3d(in_channels_ptv, hidden_dim, kernel_size=3, padding=1)
        self.oar_conv = nn.Conv3d(in_channels_oar, hidden_dim, kernel_size=3, padding=1)

        # Attention module parameters (simplified version)
        self.query_conv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1)
        self.key_conv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1)
        self.value_conv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1)

        # Output layer (fuse the attention outputs and project to final output)
        self.out_conv = nn.Conv3d(hidden_dim, out_channels, kernel_size=1)

        # Batch Normalization for stability
        self.batch_norm = nn.BatchNorm3d(hidden_dim)

    def forward(self, ct, ptv, oar):
        # Feature extraction from CT, PTV, OAR
        ct_features = F.relu(self.ct_conv(ct))
        ptv_features = F.relu(self.ptv_conv(ptv))
        oar_features = F.relu(self.oar_conv(oar))

        # Apply the first attention mechanism between CT and PTV
        ct_ptv_attention = self.cross_attention(ct_features, ptv_features)

        # Apply the second attention mechanism between the fused CT-PTV features and OAR features
        final_attention = self.cross_attention(ct_ptv_attention, oar_features)

        # Apply Batch Normalization for stability
        final_attention = self.batch_norm(final_attention)

        ct_ptv_attention = self.batch_norm(ct_ptv_attention)
        
        # Final convolution to project the output to the desired output channels
        output = self.out_conv(final_attention)
        output_ct_ptv = self.out_conv(ct_ptv_attention)
        
        return output, output_ct_ptv

    def cross_attention(self, query, key_value):
        """
        A simplified cross-attention mechanism that computes the attention between query and key-value tensors.
        """
        # Queries, keys, and values come from different modalities
        query_features = self.query_conv(query)  # Shape: [batch_size, hidden_dim, D, H, W]
        key_features = self.key_conv(key_value)  # Shape: [batch_size, hidden_dim, D, H, W]
        value_features = self.value_conv(key_value)  # Shape: [batch_size, hidden_dim, D, H, W]

        # Compute attention scores using simple dot product between query and key
        # Sum over the channels (dim=1), resulting in a shape of [batch_size, 1, D, H, W]
        attention_scores = torch.sum(query_features * key_features, dim=1, keepdim=True)

        # Normalize the attention scores using softmax (over spatial dimensions)
        attention_weights = F.softmax(attention_scores, dim=-1)  # Normalize over D, H, W

        # Compute weighted sum of values using the attention scores
        # The attention weights will be broadcasted to match the shape of value_features
        output = value_features * attention_weights

        return output


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

class Comb(nn.Module):
    def __init__(self, in_channels):
        super(Comb, self).__init__()

        # Ensure the input channels match expected encoder output
        self.in_channels = in_channels  

        # Define the gamma and beta layers to match the expected input channels
        self.gamma_layer = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.beta_layer = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        # If ptv_resized has 1 channel, expand it to match in_channels
        self.channel_expand = nn.Conv3d(1, in_channels, kernel_size=1)

    def forward(self, enc_out, ptv):
        """
        enc_out: Tensor of shape [B, in_channels, D, H, W]
        ptv: Tensor of shape [B, 1, D, H, W]
        """
        # Resize ptv to match encoder output dimensions
        ptv_resized = F.interpolate(ptv, size=enc_out.shape[2:], mode='trilinear', align_corners=False)

        # Ensure ptv_resized has the same number of channels as enc_out
        if ptv_resized.shape[1] != self.in_channels:
            ptv_resized = self.channel_expand(ptv_resized)  # Convert to in_channels

        # Pass through gamma and beta layers
        gamma = self.gamma_layer(ptv_resized)
        beta = self.beta_layer(ptv_resized)

        # Apply transformation to encoder output
        out = enc_out * (1 + gamma) + beta

        return out


class Encoder(nn.Module):
    def __init__(self, in_ch, list_ch, d_state, d_conv, expand, channel_token=False):
        super(Encoder, self).__init__()

        # Replace the existing SingleConv with ResidualMambaBlock
        self.encoder_1 = ResidualMambaBlock(1, list_ch[1], kernel_sizes=[3, 5, 7, 9], stride=1,
                                             norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU,
                                             d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')
        
        self.encoder_2 = ResidualMambaBlock(list_ch[1], list_ch[2], kernel_sizes=[3, 5, 7, 9], stride=2,
                                             norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU,
                                             d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')
        
        self.encoder_3 = ResidualMambaBlock(list_ch[2], list_ch[3], kernel_sizes=[3, 5, 7, 9], stride=2,
                                             norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU,
                                             d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')
        
        self.encoder_4 = ResidualMambaBlock(list_ch[3], list_ch[4], kernel_sizes=[3, 5, 7, 9], stride=2,
                                             norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU,
                                             d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')
        
        self.encoder_5 = ResidualMambaBlock(list_ch[4], list_ch[5], kernel_sizes=[3, 5, 7, 9], stride=2,
                                             norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU,
                                             d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')

        self.encoder_ptv = ResidualMambaBlock(1, 1, kernel_sizes=[3, 5, 7, 9], stride = 1, 
                                              norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU,
                                              d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')
        


        self.comb2 = Comb(list_ch[1])
        self.comb3 = Comb(list_ch[2])
        self.comb4 = Comb(list_ch[3])
        self.comb5 = Comb(list_ch[4])

        self.cross_att = CrossAttentionFusionModule()

    def forward(self, x):


        PTV = x[:, 0:1, :, :, :]

        OAR = x[:, 1:8, :, :, :]

        CT = x[:, 8:9, :, :, :]


        input, ct_ptv = self.cross_att(CT, PTV, OAR)

        out_encoder_1 = self.encoder_1(input)


        out_encoder_2 = self.encoder_2(self.comb2(out_encoder_1, ct_ptv))
        out_encoder_3 = self.encoder_3(self.comb3(out_encoder_2, ct_ptv))
        out_encoder_4 = self.encoder_4(self.comb4(out_encoder_3, ct_ptv))
        out_encoder_5 = self.encoder_5(self.comb5(out_encoder_4, ct_ptv))


        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5]



class AddFeatureMaps(nn.Module):
    def __init__(self, in_channels):
        super(AddFeatureMaps, self).__init__()

        # Channel Attention Module
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Global Avg Pooling
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Spatial Attention Module
        self.spatial_att = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=7, padding=3, bias=False),  # Large kernel for spatial context
            nn.Sigmoid()
        )

        # Fusion Weights (Learnable)
        self.fusion_weight1 = nn.Parameter(torch.ones(1))  # Weight for first feature map
        self.fusion_weight2 = nn.Parameter(torch.ones(1))  # Weight for second feature map

    def forward(self, F1, F2):
        """
        F1, F2: Input feature maps of shape (B, C, H, W, D)
        """
        # Channel Attention
        F1_att = self.channel_att(F1) * F1
        F2_att = self.channel_att(F2) * F2

        # Spatial Attention
        F1_att = self.spatial_att(F1_att) * F1_att
        F2_att = self.spatial_att(F2_att) * F2_att

        # Weighted Fusion
        F_fused = self.fusion_weight1 * F1_att + self.fusion_weight2 * F2_att

        return F_fused

class Decoder(nn.Module):
    def __init__(self, list_ch, d_state, d_conv, expand, channel_token=False):
        super(Decoder, self).__init__()

        self.upconv_4 = UpConv(list_ch[5], list_ch[4])

        self.decoder_conv_4 = ResidualMambaBlock(list_ch[4], list_ch[4], kernel_sizes=[3, 5, 7, 9], stride=1, norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU, d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')
       
        self.upconv_3 = UpConv(list_ch[4], list_ch[3])

        self.decoder_conv_3 = ResidualMambaBlock(list_ch[3], list_ch[3], kernel_sizes=[3, 5, 7, 9], stride=1, norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU, d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')
        
        self.upconv_2 = UpConv(list_ch[3], list_ch[2])
        
        self.decoder_conv_2 = ResidualMambaBlock(list_ch[2], list_ch[2], kernel_sizes=[3, 5, 7, 9], stride=1, norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU, d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')
        
        self.upconv_1 = UpConv(list_ch[2], list_ch[1])

        self.decoder_conv_1 = ResidualMambaBlock(list_ch[1], list_ch[1], kernel_sizes=[3, 5, 7, 9], stride=1, norm_op=nn.BatchNorm3d, nonlin=nn.LeakyReLU, d_state=d_state, d_conv=d_conv, expand=expand, se_type='CSSE3D')
        
        self.combiner_4 = AddFeatureMaps(in_channels=list_ch[4])
        self.combiner_3 = AddFeatureMaps(in_channels=list_ch[3])
        self.combiner_2 = AddFeatureMaps(in_channels=list_ch[2])
        self.combiner_1 = AddFeatureMaps(in_channels=list_ch[1])

      
    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5 = out_encoder
        
        out_decoder_4 = self.decoder_conv_4(self.combiner_4(self.upconv_4(out_encoder_5), out_encoder_4))

        out_decoder_3 = self.decoder_conv_3(self.combiner_3(self.upconv_3(out_decoder_4), out_encoder_3))
        
        out_decoder_2 = self.decoder_conv_2(self.combiner_2(self.upconv_2(out_decoder_3), out_encoder_2))
        
        out_decoder_1 = self.decoder_conv_1(self.combiner_1(self.upconv_1(out_decoder_2), out_encoder_1))
        

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

class Model_MTAS(nn.Module):
    def __init__(self, in_ch, out_ch, list_ch_A, list_ch_B, d_state, d_conv, expand, channel_token=False):
        super(Model_MTAS, self).__init__()
        self.net_A = BaseUNet(in_ch, list_ch_A, d_state, d_conv, expand, channel_token)
        self.conv_out_A = nn.Conv3d(list_ch_A[1], out_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out_net_A = self.net_A(x)
        
        output_A = self.conv_out_A(out_net_A)
        return output_A

if __name__ == '__main__':
    model = Model_MTAS(in_ch=9, out_ch=1,
                  list_ch_A=[-1, 16, 32, 64, 128, 256],
                  list_ch_B=[-1, 32, 64, 128, 256, 512],
                  d_state=16, d_conv=4, expand=2, channel_token=False)

    sample_input = torch.randn(2, 9, 128, 128, 128)


    model(sample_input)
    #a = np.sum(np.prod(v.size()) for v in model.parameters())*4e-6
    #print(a)


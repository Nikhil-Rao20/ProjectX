# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import checkpoint_seq
from timm.models.layers import trunc_normal_, DropPath
from timm.models.layers import to_2tuple
from functools import partial
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ('Conv', 'Conv2', 'LightConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus', 'GhostConv',
           'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'RepConv','CBAM_BaseConv', 'CBAM_Flatten', 'CBAM_ChannelGate', 'CBAM_ChannelPool', 'CBAM_SpatialGate', 'CBAM_Conv', 
           'ConvNeXt_LayerNorm', 'ConvNeXt_DropPath', 'ConvNeXt_GRN', 'ConvNeXt_Conv',
           'InceptionNeXt_MetaNeXtBlock',  'InceptionNeXt_Conv', 'InceptionNeXt_ConvMlp')


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    

# --------------------------------------------------------------------------------------------------------
# Modified Convolution Blocks
# --------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------
# Convolution Block Attention Module (CBAM)
# --------------------------------------------------------------------------------------------------------

class CBAM_BaseConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(CBAM_BaseConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CBAM_Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CBAM_ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM_ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            CBAM_Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                # lse_pool = cbam_logsumexp_2d(x)
                x_flat = x.view(x.size(0), x.size(1), -1)
                s, _ = torch.max(x_flat, dim=2, keepdim=True)
                lse_pool = s + (x_flat - s).exp().sum(dim=2, keepdim=True).log()

                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class CBAM_ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class CBAM_SpatialGate(nn.Module):
    def __init__(self, bn=True):
        super(CBAM_SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = CBAM_ChannelPool()
        self.spatial = CBAM_BaseConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False, bn=bn)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) 
        return x * scale

class CBAM_Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super(Conv, self).__init__()
        self.ChannelGate = CBAM_ChannelGate(c1, 16, ['avg', 'max'])
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.SpatialGate = CBAM_SpatialGate()
        self.SpatialGate_Fuse = CBAM_SpatialGate(bn=False)
    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        x_conv_out = self.act(self.bn(self.conv(x)))
        return self.conv(x_out)+x_conv_out
    def forward_fuse(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate_Fuse(x_out)
        x_conv_out = self.act(self.conv(x))
        return self.conv(x_out)+x_conv_out
        
# --------------------------------------------------------------------------------------------------------
# ConvNeXt
# --------------------------------------------------------------------------------------------------------

class ConvNeXt_LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class ConvNeXt_DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor

class ConvNeXt_GRN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, num_features))

    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        return x * self.gamma + self.beta

class ConvNeXt_Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        drop_path = 0.

        # self.dwconv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, groups=g, dilation=d)  # depthwise conv
        self.dwconv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.norm = ConvNeXt_LayerNorm(c2)
        self.pwconv1 = nn.Linear(c2, 4 * c2) 
        self.act = nn.GELU() if act else nn.Identity()
        self.grn = ConvNeXt_GRN(4 * c2)
        self.pwconv2 = nn.Linear(4 * c2, c2)
        self.drop_path = ConvNeXt_DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = x + self.drop_path(x)  # Residual connection
        return x


# --------------------------------------------------------------------------------------------------------
# InceptionNeXt_
# --------------------------------------------------------------------------------------------------------

class InceptionNeXt_MetaNeXtBlock(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer=nn.Identity,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=None,
            mlp_ratio=4,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer) if mlp_layer else nn.Identity()
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x * self.gamma.reshape(1, -1, 1, 1)
        x = self.drop_path(x) + shortcut
        return x


class InceptionNeXt_Conv(nn.Module):
    def __init__(
            self,
            c1,
            c2,
            k=3,
            s=1,
            p=None,
            d=1,
            g=1,
            num_blocks=1,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            drop_path_rate=0.0
    ):
        super().__init__()

        # Initial convolution layer
        # self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, groups=g, bias=False)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.norm = norm_layer(c2)
        self.act = act_layer()

        # Sequential MetaNeXt blocks
        drop_path_rates = [drop_path_rate] * num_blocks
        self.blocks = nn.Sequential(*[
            InceptionNeXt_MetaNeXtBlock(
                dim=c2,
                drop_path=drop_path_rates[i],
                mlp_layer=ConvMlp,
                norm_layer=norm_layer,
                act_layer=nn.GELU
            ) for i in range(num_blocks)
        ])

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.blocks(x)
        return x

# --------------------------------------------------------------------------------------------------------
# Advanced_Attention_Based_Conv_Block (Proposing)
# --------------------------------------------------------------------------------------------------------
    
class AdvancedAttentionBasedConvBlock(nn.Module):

    class DropPath_Simple(nn.Module):
        def __init__(self, drop_prob=None):
            super().__init__()
            self.drop_prob = drop_prob
        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor
        
    class Simple_Conv(nn.Module):
        default_act = nn.SiLU()  # default activation
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
            super().__init__()
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        def forward(self, x):
            return self.act(self.bn(self.conv(x)))
        def forward_fuse(self, x):
            return self.act(self.conv(x))
    class BottleneckLinear(nn.Module):
        def __init__(self, c2, expanded_dim, bottleneck_ratio=0.125):
            super().__init__()
            bottleneck_dim = int(c2 * bottleneck_ratio)
            self.bottleneck = nn.Linear(c2, bottleneck_dim, bias=False)
            self.expand = nn.Linear(bottleneck_dim, expanded_dim, bias=False)
        
        def forward(self, x):
            x = self.bottleneck(x)
            return self.expand(x)


    def __init__(self, c1,c2,k=1, s=1, p=None, g=1, d=1, act=True, ):
        super().__init__()

        conv_ratio=1.0
        expansion_ratio=1
        # norm_layer=partial(nn.LayerNorm, eps=1e-6)
        self.dwconv = Conv.Simple_Conv(c1, c2, k=k)
        # Normalization Layer
        # self.norm = norm_layer([c2])
        
        # Multi-Scale Feature Extraction
        expanded_dim = int(c2 * expansion_ratio)
        conv_channels = int(c2 * conv_ratio)

        self.fc1 = Conv.BottleneckLinear(c2, expanded_dim*2)  # For gating and multi-scale features

        # Multi-scale Convolutions
        self.conv_3x3 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3,
                                  padding=3 // 2, groups=conv_channels)
        self.conv_5x5 = nn.Conv2d(conv_channels, conv_channels, kernel_size=5,
                                  padding=5 // 2, groups=conv_channels)

        # Additive Attention
        self.att_weight_fc = Conv.BottleneckLinear(expanded_dim, 1)  # Softmax for additive attention

        # Final Linear Projection
        self.fc2 = Conv.BottleneckLinear(expanded_dim, c2)

        # Drop Path (Stochastic Depth)
        self.drop_path = Conv.DropPath_Simple(0.1)
        # self.SpatialGate = CBAM_SpatialGate()
        # self.ChannelGate = CBAM_ChannelGate(c2, 16, ['avg', 'max'])
        self.SpatialGate = SpatialAttention()
        self.ChannelGate = ChannelAttention(c2)

        # Activation
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        # Input shape: [B, C, H, W]
        # Convoluting to the output channels
        x = self.dwconv(x)
        # Permute to [B, H, W, C] for layer normalization
        x = x.permute(0, 2, 3, 1)
          # Save shortcut connection
        # x = self.norm(x)

        # Gating and Multi-Scale Features

        # Split the input tensor into two halves dynamically
        fc1_out = self.fc1(x)  # Shape: [B, H, W, C]
        split_size = fc1_out.size(-1) // 2  # Dynamically calculate split size
        g, features = torch.split(fc1_out, [split_size, fc1_out.size(-1) - split_size], dim=-1)
        
        # Process multi-scale features (Split into channels for depthwise convolution)
        shortcut = features
        features = features.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        features_3x3 = self.conv_3x3(features)
        features_5x5 = self.conv_5x5(features)

        # Combine multi-scale features
        combined_features = features+features_3x3 + features_5x5

        combined_features = self.ChannelGate(combined_features)
        combined_features = self.SpatialGate(combined_features)
        # Permute back to [B, H, W, C]
        combined_features = combined_features.permute(0, 2, 3, 1)

        # Additive Attention
        
        result = self.att_weight_fc(features.permute(0,2,3,1))
        attention_weights = torch.softmax(result, dim=-1)
        gated_output = self.act(g) * attention_weights
        # Combine features  
        x = self.fc2(gated_output + combined_features)
        # Drop path and residual connection
        x = self.drop_path(x) + shortcut
        return x.permute(0, 3, 1, 2)

# --------------------------------------------------------------------------------------------------------
# Enhanced Attention Residual Block - Used between the neck and head for better feature importance
# --------------------------------------------------------------------------------------------------------

class EnhancedAttentionResidualBlock(nn.Module):
    class CBAM(nn.Module):
        def __init__(self, channels, reduction=16, kernel_size=7):
            super().__init__()
            self.channel_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
            self.spatial_att = nn.Sequential(nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),nn.Sigmoid())
        
        def forward(self, x):
            avg_out = self.channel_att(x)
            x = x * avg_out
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            avg_out = torch.mean(x, dim=1, keepdim=True)
            spatial_out = self.spatial_att(torch.cat([max_out, avg_out], dim=1))
            return x * spatial_out
    def __init__(self, in_channels, out_channels, reduction=16, kernel_size=7, dilation=1):
        super(EnhancedAttentionResidualBlock, self).__init__()
        self.depthwise_sep_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, groups=in_channels, bias=False),
            # nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.cbam = EnhancedAttentionResidualBlock.CBAM(out_channels, reduction, kernel_size)
        self.residual_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual_connection(x)
        x = self.depthwise_sep_conv(x)
        x = self.cbam(x)
        x += residual
        return self.relu(x)
    
# --------------------------------------------------------------------------------------------------------
# End
# --------------------------------------------------------------------------------------------------------

class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]:i[0] + 1, i[1]:i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__('cv2')
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)



#####

'''
https://github.com/RuiyangJu/Fracture_Detection_Improved_YOLOv8/blob/bd3e9957183e3fcefb30213b4912a7a449bd7996/ultralytics/nn/modules/conv.py
'''

def channel_shuffle(x, groups=2):  ##shuffle channel
    # RESHAPE----->transpose------->Flatten
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out


class GAM_Attention(nn.Module):
    def __init__(self, c1, c2, group=True, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1)
        )

        self.spatial_attention = nn.Sequential(

            nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(c1, int(c1 / rate),
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(int(c1 / rate), c2,
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        # x_channel_att=channel_shuffle(x_channel_att,4) #last shuffle
        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
        out = x * x_spatial_att
        # out=channel_shuffle(out,4) #last shuffle
        return out


class GCT(nn.Module):
    def __init__(self, channels, c=2, eps=1e-5):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.eps = eps
        self.c = c

    def forward(self, x):
        y = self.avgpool(x)
        mean = y.mean(dim=1, keepdim=True)
        mean_x2 = (y ** 2).mean(dim=1, keepdim=True)
        var = mean_x2 - mean ** 2
        y_norm = (y - mean) / torch.sqrt(var + self.eps)
        y_transform = torch.exp(-(y_norm ** 2 / 2 * self.c))
        return x * y_transform.expand_as(x)


class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


class ResBlock_CBAM(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=1):
        super(ResBlock_CBAM, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )
        # self.cbam = CBAM(c1=places * self.expansion, c2=places * self.expansion, )
        self.cbam = CBAM(c1=places * self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ECAAttention(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, c1, k_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4, pos_emb=False):
        super(MHSA, self).__init__()

        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.pos = pos_emb
        if self.pos:
            self.rel_h_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, 1, int(height)]),
                                             requires_grad=True)
            self.rel_w_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, int(width), 1]),
                                             requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)  # 1,C,h*w,h*w
        c1, c2, c3, c4 = content_content.size()
        if self.pos:
            content_position = (self.rel_h_weight + self.rel_w_weight).view(1, self.heads, C // self.heads, -1).permute(
                0, 1, 3, 2)  # 1,4,1024,64

            content_position = torch.matmul(content_position, q)  # ([1, 4, 1024, 256])
            content_position = content_position if (
                    content_content.shape == content_position.shape) else content_position[:, :, :c3, ]
            assert (content_content.shape == content_position.shape)
            energy = content_content + content_position
        else:
            energy = content_content
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))  # 1,4,256,64
        out = out.view(n_batch, C, width, height)
        return out


import torch.nn.functional as F
from timm.layers.create_act import create_act_layer, get_act_layer
from timm.layers.helpers import make_divisible
from timm.layers.mlp import ConvMlp
from timm.layers.norm import LayerNorm2d


class GlobalContext(nn.Module):

    def __init__(self, channels, use_attn=True, fuse_add=False, fuse_scale=True, init_last_zero=False,
                 rd_ratio=1. / 8, rd_channels=None, rd_divisor=1, act_layer=nn.ReLU, gate_layer='sigmoid'):
        super(GlobalContext, self).__init__()
        act_layer = get_act_layer(act_layer)

        self.conv_attn = nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None

        if rd_channels is None:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        if fuse_add:
            self.mlp_add = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
        else:
            self.mlp_add = None
        if fuse_scale:
            self.mlp_scale = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
        else:
            self.mlp_scale = None

        self.gate = create_act_layer(gate_layer)
        self.init_last_zero = init_last_zero
        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_attn is not None:
            nn.init.kaiming_normal_(self.conv_attn.weight, mode='fan_in', nonlinearity='relu')
        if self.mlp_add is not None:
            nn.init.zeros_(self.mlp_add.fc2.weight)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, H * W)  # (B, 1, H * W)
            attn = F.softmax(attn, dim=-1).unsqueeze(3)  # (B, 1, H * W, 1)
            context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
            context = context.view(B, C, 1, 1)
        else:
            context = x.mean(dim=(2, 3), keepdim=True)

        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)
        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x

        return x


from timm.layers.create_conv2d import create_conv2d


class GatherExcite(nn.Module):
    def __init__(
            self, channels, feat_size=None, extra_params=False, extent=0, use_mlp=True,
            rd_ratio=1. / 16, rd_channels=None, rd_divisor=1, add_maxpool=False,
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, gate_layer='sigmoid'):
        super(GatherExcite, self).__init__()
        self.add_maxpool = add_maxpool
        act_layer = get_act_layer(act_layer)
        self.extent = extent
        if extra_params:
            self.gather = nn.Sequential()
            if extent == 0:
                assert feat_size is not None, 'spatial feature size must be specified for global extent w/ params'
                self.gather.add_module(
                    'conv1', create_conv2d(channels, channels, kernel_size=feat_size, stride=1, depthwise=True))
                if norm_layer:
                    self.gather.add_module(f'norm1', nn.BatchNorm2d(channels))
            else:
                assert extent % 2 == 0
                num_conv = int(math.log2(extent))
                for i in range(num_conv):
                    self.gather.add_module(
                        f'conv{i + 1}',
                        create_conv2d(channels, channels, kernel_size=3, stride=2, depthwise=True))
                    if norm_layer:
                        self.gather.add_module(f'norm{i + 1}', nn.BatchNorm2d(channels))
                    if i != num_conv - 1:
                        self.gather.add_module(f'act{i + 1}', act_layer(inplace=True))
        else:
            self.gather = None
            if self.extent == 0:
                self.gk = 0
                self.gs = 0
            else:
                assert extent % 2 == 0
                self.gk = self.extent * 2 - 1
                self.gs = self.extent

        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.mlp = ConvMlp(channels, rd_channels, act_layer=act_layer) if use_mlp else nn.Identity()
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        size = x.shape[-2:]
        if self.gather is not None:
            x_ge = self.gather(x)
        else:
            if self.extent == 0:
                # global extent
                x_ge = x.mean(dim=(2, 3), keepdims=True)
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * x.amax((2, 3), keepdim=True)
            else:
                x_ge = F.avg_pool2d(
                    x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2, count_include_pad=False)
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * F.max_pool2d(x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2)
        x_ge = self.mlp(x_ge)
        if x_ge.shape[-1] != 1 or x_ge.shape[-2] != 1:
            x_ge = F.interpolate(x_ge, size=size)
        return x * self.gate(x_ge)
      
#####

# %%
# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.swin_transformer_v2 import SwinTransformerV2Block

def bchw_to_bhwc(input: torch.Tensor) -> torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, height, width, channels]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, channels, height, width]
    :return: (torch.Tensor) Output tensor of the shape [batch size, height, width, channels]
    """
    return input.permute(0, 2, 3, 1)


def bhwc_to_bchw(input: torch.Tensor) -> torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, channels, height, width]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, height, width, channels]
    :return: (torch.Tensor) Output tensor of the shape [batch size, channels, height, width]
    """
    return input.permute(0, 3, 1, 2)

class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=32, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = bchw_to_bhwc(self.proj(x))  # B H W C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel 
        print("Input_proj:{%.2f}"%(flops/1e9))
        return flops

# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=32, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, H, W , C = x.shape
        x = bhwc_to_bchw(x)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel 
        print("Output_proj:{%.2f}"%(flops/1e9))
        return flops

class DownSample(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size=4, stride=2, padding = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # Input shape is B H W C
        # Convert to B C H W
        x = bhwc_to_bchw(x)
        x = self.downsample(x)
        x = bchw_to_bhwc(x)
        # Convert back to B H W C
        return x    # Output shape is B H W C

class UpSample(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # Input shape is B H W C
        # Convert to B C H W
        x = bhwc_to_bchw(x)
        x = self.upsample(x)
        x = bchw_to_bhwc(x)
        # Convert back to B H W C
        return x    # Output shape is B H W C

# %%
if __name__ == "__main__":
    # Example usage
    x = torch.randn(1, 3, 256, 288) # (1, C, H, W)
    print(x.shape)  # Should be (1, 3, 256, 288)
    output = InputProj(in_channel=3,out_channel=32,kernel_size=3,stride=1)(x)
    print(output.shape)  # Should be (1, 256, 288, 32)
    # H and W must be divisible by window_size or appropriately padded, dim must be divisible by num_heads
    output = SwinTransformerV2Block(dim=32, input_resolution=(256, 288), num_heads=4, window_size=8, shift_size=0)(output)
    output = SwinTransformerV2Block(dim=32, input_resolution=(256, 288), num_heads=4, window_size=8, shift_size=8//2)(output)
    print(output.shape)  # Should be (1, 256, 288, 32)
    output = DownSample(in_channels=32,out_channels=64)(output)
    print(output.shape)  # Should be (1, 128, 144, 64)
    output = UpSample(in_channels=64,out_channels=32)(output)
    print(output.shape)  # Should be (1, 256, 288, 32)
    output = OutputProj(in_channel=32,out_channel=1)(output)
    print(output.shape)  # Should be (1, 32, 256, 288)
# %%

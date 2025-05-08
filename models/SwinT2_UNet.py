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
    return input.permute(0, 2, 3, 1).contiguous()


def bhwc_to_bchw(input: torch.Tensor) -> torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, channels, height, width]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, height, width, channels]
    :return: (torch.Tensor) Output tensor of the shape [batch size, channels, height, width]
    """
    return input.permute(0, 3, 1, 2).contiguous()

class InputProj(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        # Input projection
        self.proj = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride, padding=self.kernel_size//2),
            act_layer()
        )
        if norm_layer is not None:
            self.norm = norm_layer(self.out_channels)
        else:
            self.norm = None

    def forward(self, x):
        # INput shape is B C H W
        x = self.proj(x)  # B C H W
        x = bchw_to_bhwc(x) # B H W C
        if self.norm is not None:
            x = self.norm(x)
        return x # Output shape is B H W C

# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channels=32, out_channels=1, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer())
        if norm_layer is not None:
            self.norm = norm_layer(self.out_channels)
        else:
            self.norm = None

    def forward(self, x):
        # Input shape is B H W C
        x = bhwc_to_bchw(x) # Convert to B C H W
        x = self.proj(x)    # B C H W
        if self.norm is not None:
            x = self.norm(x)
        return x    # Output shape is B C H W

class DownSample(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size=4, stride=2, padding = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.downsample = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, x):
        # Input shape is B H W C
        x = bhwc_to_bchw(x) # Convert to B C H W
        x = self.downsample(x)
        x = bchw_to_bhwc(x) # Convert back to B H W C
        return x    # Output shape is B H W C

class UpSample(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.upsample = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        # Input shape is B H W C
        x = bhwc_to_bchw(x) # Convert to B C H W
        x = self.upsample(x)
        x = bchw_to_bhwc(x) # Convert back to B H W C
        return x    # Output shape is B H W C

class Encoder(nn.Module):
    def __init__(self, input_resolution = (256,288), C=32, window_sizes = [8,8,4,4], head_dim=32, n_layers=4, 
                 attn_drop=0.2, proj_drop=0.2,mlp_ratio=4.0,act_layer=nn.GELU):

        super().__init__()
        self.input_resolution = input_resolution
        self.C = C
        self.n_layers = n_layers
        self.window_sizes = window_sizes
        self.head_dim = head_dim
        self.swin_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        for i in range(n_layers):
            input_resolution = (self.input_resolution[0] // (2**(i)), self.input_resolution[1] // (2**(i)))
            dim = self.C*(2**i)
            num_heads = dim // self.head_dim
            input_channels = self.C*(2**i)
            out_channels = self.C*(2**(i+1))
            window_size = self.window_sizes[i]
            #  Swin transformer block
            self.swin_blocks.append(nn.Sequential(
                SwinTransformerV2Block(
                dim=dim, 
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                qkv_bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                mlp_ratio=mlp_ratio,
                norm_layer=nn.LayerNorm,
                act_layer=act_layer),
                SwinTransformerV2Block(
                dim=dim, 
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=window_size//2,
                qkv_bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                mlp_ratio=mlp_ratio,
                norm_layer=nn.LayerNorm,
                act_layer=act_layer)))
            # Downsample block
            self.downs.append(DownSample(in_channels=input_channels, out_channels=out_channels))

    def forward(self, x):
        # Input shape is B H W C
        skip_connections = []
        for swin_block, down in zip(self.swin_blocks, self.downs):
            x = swin_block(x)   # B H W C
            skip_connections.append(x)
            x = down(x) # B H W C
        return x, skip_connections

class Bottleneck(nn.Module):
    def __init__(self, C, input_resolution, n_layers=4,head_dim=32, window_sizes=[8,8,4,4,2], 
                 attn_drop=0.2, proj_drop=0.2,mlp_ratio=4.0,act_layer=nn.GELU):
        super().__init__()
        self.C = C
        self.input_resolution = input_resolution
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.window_sizes = window_sizes
        i = n_layers
        input_resolution = (self.input_resolution[0] // (2**(i)), self.input_resolution[1] // (2**(i)))
        dim = self.C*(2**i)
        num_heads = dim // self.head_dim
        window_size = self.window_sizes[i]
        self.bottleneck =  nn.Sequential(
                SwinTransformerV2Block(
                dim=dim, 
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                qkv_bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                mlp_ratio=mlp_ratio,
                norm_layer=nn.LayerNorm,
                act_layer=act_layer),
                SwinTransformerV2Block(
                dim=dim, 
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=window_size//2,
                qkv_bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                mlp_ratio=mlp_ratio,
                norm_layer=nn.LayerNorm,
                act_layer=act_layer))
        
    def forward(self, x):
        # Input shape is B H W C
        x = self.bottleneck(x)  # B H W C
        return x

class Decoder(nn.Module):
    def __init__(self, input_resolution = (256,288), C=32, window_sizes = [8,8,4,4], head_dim=32, n_layers=4,
                 attn_drop=0.2, proj_drop=0.2,mlp_ratio=4.0,act_layer=nn.GELU):
        super().__init__()
        self.input_resolution = input_resolution
        self.C = C
        self.n_layers = n_layers
        self.window_sizes = window_sizes
        self.head_dim = head_dim
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.swin_blocks = nn.ModuleList()

        for i in range(n_layers-1,-1,-1):
            input_resolution = (self.input_resolution[0] // (2**(i)), self.input_resolution[1] // (2**(i)))
            dim = self.C*(2**i)
            num_heads = dim // self.head_dim
            input_channels = self.C*(2**(i+1))
            out_channels = self.C*(2**(i))
            window_size = self.window_sizes[i]
            # Up blocks
            self.ups.append(UpSample(in_channels=input_channels, out_channels=out_channels))
            # conv blocks
            self.convs.append(nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0))
            #  Swin transformer block
            self.swin_blocks.append(nn.Sequential(
                SwinTransformerV2Block(
                dim=dim, 
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                qkv_bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                mlp_ratio=mlp_ratio,
                norm_layer=nn.LayerNorm,
                act_layer=act_layer),
                SwinTransformerV2Block(
                dim=dim, 
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=window_size//2,
                qkv_bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                mlp_ratio=mlp_ratio,
                norm_layer=nn.LayerNorm,
                act_layer=act_layer)))

    def forward(self, x,skip_connections):
        # Input shape is B H W C
        for up, conv, swin_block, skip in zip(self.ups, self.convs, self.swin_blocks, skip_connections[::-1]):
            x = up(x)   # B H W C
            x = torch.cat((x, skip), dim=-1).contiguous()    # Concatenate along the channel dimension, that is -c
            x = bchw_to_bhwc(conv(bhwc_to_bchw(x))) # B H W C
            x = swin_block(x)
        return x

class SwinT2UNet(nn.Module):
    '''
    Swin Transformer V2 UNet
    This architecture is implemented followed by the paper "Swin Transformer V2: Scaling Up Capacity and Resolution"
    Parameters:
    - input_resolution (tuple): Input resolution of the image.
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - C (int): Number of channels in the intermediate layers.
    - n_layers (int): Number of layers in the encoder and decoder.
    - window_sizes (list): List of window sizes for each layer.
    - head_dim (int): Dimension of each head in the multi-head attention.
    - hard_enforce_stations (bool): If True, enforces station values in the output. Technically, the output will have station values at the station locations.
    '''

    def __init__(self, input_resolution=(256,288), in_channels=3, out_channels=1, C=32, n_layers=4, attn_drop=0.2, proj_drop=0.2,mlp_ratio=4.0,act_layer=nn.GELU,
                 window_sizes=[8,8,4,4], head_dim=32,hard_enforce_stations=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.C = C
        self.n_layers = n_layers
        self.window_sizes = window_sizes
        self.head_dim = head_dim
        self.input_proj = InputProj(in_channels, C,act_layer=act_layer)
        self.hard_enforce_stations = hard_enforce_stations
        self.encoder = Encoder(input_resolution=self.input_resolution, C=self.C, window_sizes=self.window_sizes, head_dim=self.head_dim, n_layers=self.n_layers,
                               attn_drop=attn_drop, proj_drop=proj_drop,mlp_ratio=mlp_ratio,act_layer=act_layer)
        self.bottleneck = Bottleneck(C=self.C, input_resolution=self.input_resolution, n_layers=self.n_layers, window_sizes=self.window_sizes, head_dim=self.head_dim,
                                     attn_drop=attn_drop, proj_drop=proj_drop,mlp_ratio=mlp_ratio,act_layer=act_layer)
        self.decoder = Decoder(input_resolution=self.input_resolution, C=self.C, window_sizes=self.window_sizes, head_dim=self.head_dim, n_layers=self.n_layers,
                               attn_drop=attn_drop, proj_drop=proj_drop,mlp_ratio=mlp_ratio,act_layer=act_layer)
        self.output_proj = OutputProj(in_channels=self.C, out_channels=self.out_channels)
        
    def forward(self, x):
        # Input shape is B C H W
        if self.hard_enforce_stations:
            station_values = x[:, 0, ...].unsqueeze(1)  # [B, 1, H, W]
            station_mask = x[:, -1, ...].unsqueeze(1)  # [B, 1, H, W]
        x = self.input_proj(x)  # B H W C
        x, skip_connections = self.encoder(x)  # B H W C
        x = self.bottleneck(x)  # B H W C
        x = self.decoder(x, skip_connections)  # B H W C
        x = self.output_proj(x)  # B C H W
        if self.hard_enforce_stations:
            x = station_mask * station_values + (1-station_mask)*x
        return x    # Output shape is B C H W

# %%
if __name__ == "__main__":
    from util import initialize_weights_xavier,initialize_weights_he
    # Example usage
    input_resolution = (256, 288)
    in_channels = 3
    out_channels = 1
    C = 32
    n_layers = 4
    window_sizes = [8, 8, 4, 4, 2]
    head_dim = 32
    attn_drop = 0.2
    proj_drop = 0.2
    mlp_ratio = 4.0
    act_layer = nn.GELU
    seed = 42
    model = SwinT2UNet(input_resolution=input_resolution, 
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        C=C, n_layers=n_layers, 
                        window_sizes=window_sizes,
                        head_dim=head_dim,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        mlp_ratio=mlp_ratio,
                        act_layer=act_layer,
                        hard_enforce_stations=True)
    if act_layer == nn.GELU:
        initialize_weights_xavier(model,seed = seed)
    elif act_layer == nn.ReLU:
        initialize_weights_he(model,seed = seed)
    print(model)  # Print the model architecture
    # print the total number of parameters in the model
    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    # Create a random input tensor
    x = torch.randn(1, in_channels, input_resolution[0], input_resolution[1])  # (batch_size, channels, height, width)
    print(x.shape)  # Should be (1, 3, 256, 288)
    output = model(x)
    print(output.shape)  # Should be (1, 1, 256, 288)
    # Print weights of the first conv layer
    first_conv_layer = model.input_proj.proj[0]  # First nn.Conv2d inside the first block
    print("Weights of the first Conv2d layer:")
    print(first_conv_layer.weight.data)
# %%

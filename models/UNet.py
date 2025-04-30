# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

# %%
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1,dropout_prob=0.2,drop_path_prob=0.1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob)
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

    def forward(self, x):
        out1 = self.block(x)
        out1 = self.drop_path(out1)
        out2 = self.conv11(x)
        out = out1 + out2
        return out
    
class Encoder(nn.Module):
    def __init__(self,in_channels, C, n_layers=4):
        super().__init__()
        self.in_channels = in_channels
        self.C = C
        self.n_layers = n_layers
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        for i in range(n_layers):
            if i ==0:
                in_channels = self.in_channels
                out_channels = self.C
            else:
                in_channels = self.C*(2**(i-1))
                out_channels = self.C*(2**i)
            self.blocks.append(ConvBlock(in_channels, out_channels))
            self.downs.append(nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))

    def forward(self, x):
        skip_connections = []
        for block, down in zip(self.blocks, self.downs):
            x = block(x)
            #print("x convblock shape", x.shape)
            skip_connections.append(x)
            x = down(x)
            #print("x down shape", x.shape)
        return x, skip_connections

class Bottleneck(nn.Module):
    def __init__(self, C, n_layers=4):
        super().__init__()
        self.C = C
        self.n_layers = n_layers
        in_channels = self.C*(2**(n_layers-1))
        out_channels = self.C*(2**n_layers)
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.block(x)
        #print("x bottleneck shape", x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels, C, n_layers=4):
        super().__init__()
        self.out_channels = out_channels
        self.C = C
        self.n_layers = n_layers
        self.blocks = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i in range(n_layers-1, -1, -1):
            in_channels = self.C*(2**(i+1))
            out_channels = self.C*(2**(i))
            self.ups.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            self.blocks.append(ConvBlock(in_channels, out_channels, strides=1))
        self.blocks.append(nn.Conv2d(self.C, self.out_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x, skip_connections):
        for up, block, skip in zip(self.ups, self.blocks[:-1], skip_connections[::-1]):
            x = up(x)
            #print("x up shape", x.shape)
            x = torch.cat((x, skip), dim=1)
            #print("x cat shape", x.shape)
            x = block(x)
            #print("x block shape", x.shape)
        x = self.blocks[-1](x)
        #print("x final shape", x.shape)
        return x

class UNet(nn.Module):
    '''
    UNet architecture for image reconstruction tasks.
    This model consists of an encoder, a bottleneck, and a decoder.
    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - C (int): Number of channels (or) dimensions in the intermediate layers.
    - n_layers (int): Number of convolutional layers in the network.
    - kernel (tuple): Size of the convolutional kernel. We used 3*3 kernel.
    - hard_enforce_stations (bool): If True, enforces station values in the output. Technically, the output will have station values at the staiton locations.
    Returns:
    - output (Tensor): Output tensor after passing through the network.
    The entire architectue is based on the paper "Uformer: A General U-Shaped Transformer for Image Restoration"
    The corresponding code is available at "https://github.com/ZhendongWang6/Uformer"
    In the encoder, the Channels will double after each ConvBlock, while the spatial dimensions will be halved after each downsampling layer.
    In the decoder, the spatial dimensions will be doubled after each upsampling layer, then the channels will double by concatnation, while the channels will be halved after each ConvBlock.

    '''
    def __init__(self, in_channels=3, out_channels=1, C=32, n_layers=4,hard_enforce_stations=False):
        super(UNet, self).__init__()
        self.hard_enforce_stations = hard_enforce_stations
        self.encoder = Encoder(in_channels, C, n_layers)
        self.bottleneck = Bottleneck(C, n_layers)
        self.decoder = Decoder(out_channels, C, n_layers)

    def forward(self, x):
        if self.hard_enforce_stations:
            station_values = x[:, 0, ...].unsqueeze(1)  # [B, 1, H, W]
            station_mask = x[:, -1, ...].unsqueeze(1)  # [B, 1, H, W]
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        if self.hard_enforce_stations:
            x = station_mask * station_values + (1-station_mask)*x
        return x
    
# %%    
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1, C=32, n_layers=4,hard_enforce_stations=True)
    print(model)
    # print the total number of parameters in the model
    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    x = torch.randn(1, 3, 256, 288)  # Example input
    output = model(x)
    print("Output shape:", output.shape)  # Should be (1, 1, 256, 256)
    '''
    Already examined the model architecture, intermediate outputs shape and the final output shape.
    ''' 
# %%

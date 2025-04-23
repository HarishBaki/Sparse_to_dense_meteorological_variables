# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels, feature_channels=[64, 128, 256], dropout_prob=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        for out_channels in feature_channels:
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_prob)
            )
            self.blocks.append(block)
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

    def forward(self, x):
        skip_connections = []

        for block, pool in zip(self.blocks, self.pools):
            identity = x
            out = block(x)
            if identity.shape == out.shape:
                out = out + identity  # residual connection
            skip_connections.append(out)
            x = pool(out)

        return x, skip_connections

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob)
        )

    def forward(self, x):
        identity = x
        out = self.block(x)
        if identity.shape == out.shape:
            out = out + identity  # Residual connection
        return out
    
class Decoder(nn.Module):
    def __init__(self, feature_channels=[256, 128, 64], dropout_prob=0.1):
        super().__init__()
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()

        for i in range(len(feature_channels) - 1):
            in_ch = feature_channels[i]
            out_ch = feature_channels[i + 1]

            self.ups.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )

            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=dropout_prob)
                )
            )

    def forward(self, x, skip_connections):
        for i in range(len(self.ups)):
            x = self.ups[i](x)
            skip = skip_connections[-(i + 1)]  # reverse order
            x = torch.cat([x, skip], dim=1)

            identity = x
            out = self.blocks[i](x)
            if out.shape == identity.shape:
                x = out + identity  # residual connection
            else:
                x = out  # if shapes mismatch (shouldn't happen if matched carefully)

        return x

class GoogleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_channels=[64, 128, 256], dropout_prob=0.1):
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels, feature_channels=feature_channels, dropout_prob=dropout_prob)
        
        self.bottleneck = Bottleneck(
            in_channels=feature_channels[-1],
            out_channels=feature_channels[-1] * 2,
            dropout_prob=dropout_prob
        )

        # Decoder takes reversed feature channels from bottleneck → first encoder layer
        decoder_channels = [feature_channels[-1] * 2] + feature_channels[::-1]
        self.decoder = Decoder(feature_channels=decoder_channels, dropout_prob=dropout_prob)

        # Final projection to desired output channels
        self.final_conv = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        x = self.final_conv(x)
        return x
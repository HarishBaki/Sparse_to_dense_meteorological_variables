# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
class DCNN(nn.Module):
    '''
    This architecture is implimented followed by the paper "Global field reconstruction from sparse sensors with Voronoi tessellation-assisted deep learning"
    The architecture available at "https://github.com/kfukami/Voronoi-CNN"
    Deep Convolutional Neural Network (DCNN) for image processing tasks.
    This model consists of multiple convolutional layers followed by ReLU activations.
    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - C (int): Number of channels in the intermediate layers.
    - kernel (tuple): Size of the convolutional kernel.
    - final_kernel (tuple): Size of the final convolutional kernel.
    - n_layers (int): Number of convolutional layers in the network.
    - hard_enforce_stations (bool): If True, enforces station values in the output. Technically, the output will have station values at the staiton locations.
    Returns:
    - output (Tensor): Output tensor after passing through the network.
    '''
    def __init__(self, in_channels, out_channels, C=48,kernel=(7,7),final_kernel=(3,3),n_layers=7,act_layer=nn.ReLU,hard_enforce_stations=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.C = C
        self.kernel = kernel
        self.final_kernel = final_kernel    
        self.n_layers = n_layers
        self.blocks = nn.ModuleList()
        self.hard_enforce_stations = hard_enforce_stations
        for i in range(n_layers):
            if i == 0:
                in_channels = self.in_channels
                out_channels = self.C
            else:
                in_channels = self.C
                out_channels = self.C
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=(kernel[0] // 2, kernel[1] // 2)),
                act_layer()
            ))
        self.final_conv = nn.Conv2d(C, self.out_channels, kernel_size=self.final_kernel, padding=1)

    def forward(self, x):
        if self.hard_enforce_stations:
            station_values = x[:, 0, ...].unsqueeze(1)  # [B, 1, H, W]
            station_mask = x[:, -1, ...].unsqueeze(1)  # [B, 1, H, W]
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        if self.hard_enforce_stations:
            x = station_mask * station_values + (1-station_mask)*x
        return x

# %%
if __name__ == "__main__":
    from util import initialize_weights_xavier,initialize_weights_he
    # Example usage
    act_layer = nn.GELU
    seed = 42
    model = DCNN(in_channels=3, out_channels=1, C=48, kernel=(7, 7),final_kernel=(3,3), n_layers=7,act_layer=act_layer,hard_enforce_stations=True)
    if act_layer == nn.GELU:
        initialize_weights_xavier(model,seed = seed)
    elif act_layer == nn.ReLU:
        initialize_weights_he(model,seed = seed)
    print(model)  # Print the model architecture
    # print the total number of parameters in the model
    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    x = torch.randn(1, 3, 256, 288)  
    output = model(x)
    print(output.shape)
    '''
    Already examined the model architecture, intermediate outputs shape and the final output shape.
    The shape is retained from from end to end.
    '''  
    # Print weights of the first conv layer
    first_conv_layer = model.blocks[0][0]  # First nn.Conv2d inside the first block
    print("Weights of the first Conv2d layer:")
    print(first_conv_layer.weight.data)
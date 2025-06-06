import torch
import torch.nn as nn
from einops import rearrange
import math

class HFE(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.spectral_conv = nn.Sequential(nn.Conv1d(dim, hidden_dim * 2, kernel_size=3, padding=1),
                                           act_layer()) 

        self.dwconv2 = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim * 2, groups=hidden_dim*2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=1, stride=1),
            act_layer())
        self.linear3 = nn.Sequential(nn.Linear(hidden_dim*4, dim))

        self.weight = nn.Parameter(torch.randn(2))
        self.linear4 = nn.Sequential(nn.Linear(hidden_dim*2, dim))
    def forward(self, x):  
        x_init = x

        bs, c, h, w = x.size()
        x1, x2, = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=h, w=w)
        x = self.linear1(x)   
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=h, w=w)  

        local_features = self.dwconv2(x)  
        global_features = self.global_avg_pool(x)   
        global_features = global_features.expand_as(local_features)  

        fused_features = torch.cat([local_features, global_features], dim=1)  # 
        B, C, H, W = fused_features.shape
        fused_features = fused_features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        fused_features = self.linear3(fused_features)
        fused_features = fused_features.view(B, H, W, 128).permute(0, 3, 1, 2)

        return fused_features + x_init


if __name__ == '__main__':

    dim = 128  # Dimension of input features

    hfe = HFE(dim)

    # Generate a random input tensor
    B = 1  # Batch size
    H = 64  # Height of the feature map
    W = 64  # Width of the feature map
    C = dim  # Number of channels

    input = torch.randn(B, C, H, W)

    # Forward pass
    output = hfe(input)

    # Print input and output shapes
    print(input.size())
    print(output.size())
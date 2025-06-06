import torch
import torch.nn as nn

        
class HCA(nn.Module):
    def __init__(self, in_channels, reduction=8, kernel_size=3, normal_residual=True):
        super(HCA, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=kernel_size, 
                               padding=kernel_size//2, bias=False)  
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)  
        self.sigmoid = nn.Sigmoid()
        self.normal_residual = normal_residual

        self.w1 = nn.Parameter(torch.ones(1))  
        self.w2 = nn.Parameter(torch.ones(1)) 

    def forward(self, x):
        y = self.global_avg_pool(x) 
        y = self.conv1(y)  
        y = self.relu(y)
        y = self.conv2(y)  
        y = self.sigmoid(y)  
        if self.normal_residual:
            return x * y + x  
        else:
            return self.w1 * (x * y) + self.w2 * x


    
if __name__ == '__main__':

    dim = 128  

    seb = HCA(dim)
    # Generate a random input tensor
    B = 1  # Batch size
    H = 64  # Height of the feature map
    W = 64  # Width of the feature map
    C = dim  # Number of channels

    input = torch.randn(B, C, H, W)
    # Forward pass
    output = seb(input)

    # Print input and output shapes
    print(input.size())
    print(output.size())

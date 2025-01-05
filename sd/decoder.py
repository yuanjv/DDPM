import torch
from torch import nn
from torch.nn import functional as F 
from attention import SelfAttention

class VAE_ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()
        self.gn0=nn.GroupNorm(32,in_channels)
        
        self.conv0=nn.Conv2d(
            in_channels, out_channels, kernel_size=3,padding=1
        )

        self.gn1=nn.GroupNorm(32,out_channels)

        
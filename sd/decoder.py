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
        
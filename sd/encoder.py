import torch
from torch import nn
from torch.nn import functional as F 
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

CHANNEL=128

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        """
        less pix, more info/pix
        """
        super().__init__(
            #bs, channel, w, h
            nn.Conv2d(3,CHANNEL,kernel_size=3,padding=1),
            
            VAE_ResidualBlock(CHANNEL,CHANNEL),
            VAE_ResidualBlock(CHANNEL,CHANNEL),
            
            #w/2,h/2
            nn.Conv2d(CHANNEL,CHANNEL,kernel_size=3,stride=2,padding=0),
            
            #channel*2
            VAE_ResidualBlock(CHANNEL,CHANNEL*2),

            VAE_ResidualBlock(CHANNEL*2,CHANNEL*2)
        )

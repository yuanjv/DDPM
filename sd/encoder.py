import torch
from torch import nn
from torch.nn import functional as F 
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

CHANNEL=128
OUT_CHANNEL=8

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
            
            #bs, channel, w/2, h/2
            nn.Conv2d(CHANNEL,CHANNEL,kernel_size=3,stride=2,padding=0),
            
            #bs, channel*2, w/2, h/2
            VAE_ResidualBlock(CHANNEL,CHANNEL*2),

            VAE_ResidualBlock(CHANNEL*2,CHANNEL*2),

            #bs, c*2, h/4, w/4
            nn.Conv2d(CHANNEL*2,CHANNEL*2,kernel_size=3,stride=2,padding=0),

            #bs,c*4, w/4, h/4
            VAE_ResidualBlock(CHANNEL*2,CHANNEL*4),

            VAE_ResidualBlock(CHANNEL*4,CHANNEL*4),

            #bs, c*4, w/8, h/8
            nn.Conv2d(CHANNEL*4,CHANNEL*4,kernel_size=3,stride=2,padding=0),

            VAE_ResidualBlock(CHANNEL*4,CHANNEL*4),
            VAE_ResidualBlock(CHANNEL*4,CHANNEL*4),
            VAE_ResidualBlock(CHANNEL*4,CHANNEL*4),


            VAE_AttentionBlock(CHANNEL*4),
            VAE_ResidualBlock(CHANNEL*4,CHANNEL*4),

            nn.GroupNorm(32,CHANNEL*4),

            nn.SiLU(),

            #bs, out_channel, w/8, h/8
            nn.Conv2d(CHANNEL*4,OUT_CHANNEL,kernel_size=3,padding=1),

            nn.Conv2d(OUT_CHANNEL,OUT_CHANNEL,kernel_size=3,padding=1),
        )
    
    def forward(
        self,
        x:torch.Tensor,
        noise:torch.Tensor
    ) -> torch.Tensor:
        # x: bs, c, h, w
        # noise: bs, oc, h/8, w/8

        for m in self:
            if getattr(m, "stride",None)==(2,2):
                #left,right✅, top, bottom✅
                x=F.pad(x,(0,1,0,1))
            x=m(x)
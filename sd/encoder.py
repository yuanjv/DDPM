import torch
from torch import nn
from torch.nn import functional as F 
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

CHANNEL=128
OUT_CHANNEL=8

'''
    old old notes from 2 years ago
    https://colab.research.google.com/drive/19GUkNMQ8O_ZWMr-bylUjhVMBEMW7GQE7?usp=sharing
    based on https://github.com/mrdbourke/pytorch-deep-learning
        kernel_size=3,#3*3
        stride=1,#move 1 pix at a time --> output the same size
        padding=1#add 2 to the h & w (1 left 1 right) to increase the output size by 2
'''


class VAE_Encoder(nn.Sequential):
    def __init__(self,in_channels:int=3): #rgb
        """
        less pix, more info/pix

        channel ++, size --
        """
        super().__init__(
            #bs, channel, w, h
            nn.Conv2d(in_channels,CHANNEL,kernel_size=3,padding=1),
            
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

            nn.GroupNorm(32,CHANNEL*4), #32 works great since 1st paper https://youtu.be/l_3zj6HeWUE

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
        
        #bs, oc, h/8, w/8 -> 2 * (bs, oc/2, h/8, w/8)
        mu,log_var=torch.chunk(x,2,dim=1)

        #linit range -> var -> std
        #N(mu, std^2)
        std=torch.clamp(log_var,-30,20).exp().sqrt()

        x=mu+std*noise

        return x*0.18215
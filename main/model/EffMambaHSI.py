import math
import torch
from torch import nn
from mamba_ssm import Mamba
from .HCA import HCA
from .HFE import HFE

class DualGroupMamaba(nn.Module):
    def __init__(self, channels=128, num_groups=4, group_num=4, use_proj=True,use_residual=True):
        super(DualGroupMamaba, self).__init__()
        self.use_residual = use_residual
        self.num_groups = num_groups
        self.group_size = channels // num_groups  
        self.use_proj = use_proj
        self.local_mamba = Mamba(
            d_model=num_groups, 
            d_state=16,
            d_conv=4,
            expand=2
        ) 
        self.global_mamba = Mamba(
            d_model=self.group_size,  
            d_state=16,
            d_conv=4,
            expand=2
        )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )

 
    def forward(self, x):
        x_int = x
        B, C, H, W = x.shape 
        x = x.permute(0, 2, 3, 1).view(B*H*W, C) 
        x_groups = x.view(B*H*W, self.group_size, self.num_groups) 
        local_out = self.local_mamba(x_groups) + torch.flip(self.local_mamba(torch.flip(x_groups, dims=[1])),dims=[1])
        local_out = local_out.transpose(1, 2) 
        global_out = self.global_mamba(local_out) + torch.flip(self.global_mamba(torch.flip(local_out, dims=[1])),dims=[1])
        final_out = global_out.view(B, H, W, C).permute(0, 3, 1, 2)
        if self.use_proj:
            final_out = self.proj(final_out)

        if self.use_residual:
            return final_out + x_int
        else:
            return final_out


class BidirMamba(nn.Module):
    def __init__(self,channels,use_residual=True,group_num=4,use_proj=True):
        super(BidirMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(d_model=channels,  
                           d_state=16,  
                           d_conv=4,  
                           expand=2,  
                           )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )

    def forward(self,x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B,H,W,C = x_re.shape
        x_flat = x_re.view(1,-1, C)
        x_flat = self.mamba(x_flat) +  torch.flip(self.mamba(torch.flip(x_flat, dims=[1])), dims=[1])
        x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_recon = self.proj(x_recon)
        if self.use_residual:
            return x_recon + x
        else:
            return x_recon


class BothMamba(nn.Module):
    def __init__(self,channels,token_num,use_residual,group_num=4,use_att=True):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = BidirMamba(channels,use_residual=use_residual,group_num=group_num)
        self.spe_mamba = DualGroupMamaba(channels=channels,group_num=group_num)
        

    def forward(self,x):
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)
        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = spa_x * weights[0] + spe_x * weights[1]
        else:
            fusion_x = spa_x + spe_x
        if self.use_residual:
            return fusion_x + x
        else:
            return fusion_x


class EffMambaHSI(nn.Module):
    def __init__(self,in_channels=128,hidden_dim=64,num_classes=10,use_residual=True,token_num=4,group_num=4,use_att=True):
        super(EffMambaHSI, self).__init__()

        self.patch_embedding = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=hidden_dim,kernel_size=1,stride=1,padding=0),  
                                             nn.GroupNorm(group_num,hidden_dim),            
                                             nn.SiLU())

        self.mamba = nn.Sequential(BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att),
                                       HCA(in_channels=128,reduction=32,normal_residual=False),
                                       nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                       BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att),
                                       HCA(in_channels=128,reduction=32,normal_residual=False),
                                       nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                       BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att),
                           
                                       )


        self.cls_head = nn.Sequential(nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
                                      nn.GroupNorm(group_num,128),                              
                                      nn.SiLU(),
                                      nn.Conv2d(in_channels=128,out_channels=num_classes,kernel_size=1,stride=1,padding=0))

        self.hfe = HFE(dim=128)

    def forward(self,x):

        x = self.patch_embedding(x)
        x = self.mamba(x)
        x = self.hfe(x)
        logits = self.cls_head(x)
        return logits


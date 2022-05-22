
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numbers
import math
from timm.models.layers import trunc_normal_
from functools import partial




class DTN(nn.Module):
    def __init__(self, num_features, eps=1e-5, use_local_init=True, locality_strength=1., \
                affine=True, group_num=1, momentum=0.9,
                resolution=(14,14), patch_size=14, only_var=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.resolution = resolution
        self.num_heads = group_num
        self.only_var = only_var
        if resolution[0] > 14: # can also set to 7
            self.patch_pool_size = resolution[0] // patch_size
            self.patch_size = patch_size
        else:
            self.patch_pool_size = 1
            self.patch_size = resolution[0]


        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if not only_var:
            self.mean_norm_weight = nn.Parameter(torch.zeros(group_num))
        self.var_norm_weight = nn.Parameter(torch.zeros(group_num))
        if self.patch_pool_size > 1:
            self.avg_pool = nn.AvgPool2d(self.patch_pool_size)
        self.pos_proj = nn.Linear(3, self.num_heads)

        #initialization
        self.apply(self.reset_parameters)
        if use_local_init:
            self.local_init(locality_strength=locality_strength)

    def reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):

        B, T, C = input.shape
        #l2 normalize for stability
        gn_input = input.contiguous().reshape(B, T, self.num_heads, -1)
        moment2 = torch.mean(gn_input * gn_input, dim=3, keepdim=True)
        input = gn_input / torch.sqrt(moment2 + self.eps)
        input = input.reshape(B,T,C)

        if not self.only_var:
            mean_norm_weight = torch.sigmoid(self.mean_norm_weight)
        var_norm_weight = torch.sigmoid(self.var_norm_weight)

        if not hasattr(self, 'rel_indices') or self.rel_indices.size(1)!=self.patch_size*self.patch_size:
            self.get_rel_indices(self.patch_size*self.patch_size)
        pos_score = self.rel_indices.expand(B, -1, -1,-1)
        pos_score = self.pos_proj(pos_score).permute(0,3,1,2)
        pos_score = pos_score.softmax(dim=-2) #B,H,P,P

        #input size B, T, C
        #statistics of LN
        mean_ln = input.mean(dim=2)
        var_ln  = input.var(dim=2)

        #statistics of position-aware IN
        H, W = self.resolution
        assert T == H * W, "input feature {} has wrong size {}".format(input.shape, self.resolution)
        input_ = input.view(B, H, W, C).permute(0,3,1,2).contiguous()
        
        if self.patch_pool_size > 1:
            input_ = self.avg_pool(input_)
        #statistics of position-aware IN
        input_ = input_.view(B, self.num_heads, C//self.num_heads, -1)
        mean_rppn = torch.matmul(input_, pos_score)
        mean2_rppn = torch.matmul(input_ * input_, pos_score)
        var_rppn = mean2_rppn - mean_rppn * mean_rppn

        if self.patch_pool_size > 1:
            mean_rppn = mean_rppn.view(B, self.num_heads, C//self.num_heads, self.patch_size, self.patch_size)
            mean_rppn = mean_rppn.unsqueeze(-1).repeat(1,1,1,1,1,self.patch_pool_size*self.patch_pool_size)
            mean_rppn = mean_rppn.view(B, self.num_heads, C//self.num_heads,self.patch_size, self.patch_size,
                                        self.patch_pool_size, self.patch_pool_size)
            mean_rppn = mean_rppn.permute(0,1,2,3,5,4,6).contiguous()
            mean_rppn = mean_rppn.view(B, self.num_heads, C//self.num_heads, -1)

            var_rppn = var_rppn.view(B, self.num_heads, C//self.num_heads, self.patch_size, self.patch_size)
            var_rppn = var_rppn.unsqueeze(-1).repeat(1,1,1,1,1,self.patch_pool_size*self.patch_pool_size)
            var_rppn = var_rppn.view(B, self.num_heads, C//self.num_heads,self.patch_size, self.patch_size,
                                        self.patch_pool_size, self.patch_pool_size)
            var_rppn = var_rppn.permute(0,1,2,3,5,4,6).contiguous()
            var_rppn = var_rppn.view(B, self.num_heads, C//self.num_heads, -1)

        #statistics of DTN
        if not self.only_var:
            mean_rppn = (1. - mean_norm_weight.view(1,self.num_heads,1,1)) * mean_rppn 
            mean_rppn += mean_norm_weight.view(1,self.num_heads,1,1) * mean_ln.view(B,1,1,-1)
            mean_rppn = mean_rppn.view(B,C,T).permute(0,2,1)

        var_rppn = (1. - var_norm_weight.view(1,self.num_heads,1,1)) * var_rppn
        var_rppn += var_norm_weight.view(1,self.num_heads,1,1) * var_ln.view(B,1,1,-1)
        var_rppn = var_rppn.view(B,C,-1).permute(0,2,1)
        # standardization
        if not self.only_var:
            input = (input - mean_rppn) / torch.sqrt(var_rppn + self.eps)
        else:
            input = (input - mean_ln.view(B,T,1)) / torch.sqrt(var_rppn + self.eps)

        return input * self.weight.view(1,1,C) + self.bias.view(1,1,C)


    def local_init(self, locality_strength=1.):

        locality_distance = 1 #max(1,1/locality_strength**.5)
        kernel_size = int(self.num_heads**.5)
        center = (kernel_size-1)/2 if kernel_size%2==0 else kernel_size//2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1+kernel_size*h2
                self.pos_proj.weight.data[position,2] = -1
                self.pos_proj.weight.data[position,1] = 2*(h1-center)*locality_distance
                self.pos_proj.weight.data[position,0] = 2*(h2-center)*locality_distance
        self.pos_proj.weight.data *= locality_strength

    def get_rel_indices(self, num_patches):
        img_size = int(num_patches**.5)
        rel_indices   = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1,-1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size,img_size)
        indy = ind.repeat_interleave(img_size,dim=0).repeat_interleave(img_size,dim=1)
        indd = indx**2 + indy**2
        rel_indices[:,:,:,2] = indd.unsqueeze(0)
        rel_indices[:,:,:,1] = indy.unsqueeze(0)
        rel_indices[:,:,:,0] = indx.unsqueeze(0)
        device = self.weight.device
        self.rel_indices = rel_indices.to(device)


if __name__ == '__main__':
    DTN = DTN(num_features=384,group_num=6,resolution=(56,56))
    #print(DTN.named_parameters())
    x = torch.rand(2,196*16,384)
    y = DTN(x)
    print(y.size())

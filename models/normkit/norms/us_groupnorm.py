#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class GroupNorm_(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(GroupNorm_, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.group_feature = num_channels // num_groups
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.gn = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        B, T, C = input.shape
        input = input.permute(0,2,1).contiguous()
        input = self.gn(input).permute(0,2,1).contiguous()
        input = self.weight.view(1,1,C) * input + self.bias.view(1,1,C)

        return input


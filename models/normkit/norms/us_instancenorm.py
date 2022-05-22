#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numbers




class BatchNorm_(nn.Module):

    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(BatchNorm_, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        B, T, C = input.shape

        input = input.permute(0,2,1).contiguous()

        input = self.bn(input)

        input = input.permute(0,2,1).contiguous()
        input = self.weight.view(1,1,C) * input + self.bias.view(1,1,C)
        #y = y.permute(0, 2, 1).contiguous()

        return input

    def extra_repr(self):
        return '{num_features}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)

class InsNorm(nn.Module):

    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(InsNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, pad_mask=None, is_encoder=False):
        B, T, C = input.shape
        input = input.permute(0, 2, 1).contiguous()
        input = input.view(input.size(0), self.num_features, -1)

        mean_in = input.mean(-1, keepdim=True)
        var_in = input.var(-1, keepdim=True)

        x = input
        x = (x-mean_in) / (var_in+self.eps).sqrt()
        x = x.view(input.size(0), C, -1)
        y = self.weight.view(1,C,1) * x + self.bias.view(1,C,1)
        y = y.permute(0, 2, 1).contiguous()

        return y

    def extra_repr(self):
        return '{num_features}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)

#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from ast import Raise
from models.normkit.norms.us_layernorm import LayerNorm
from models.normkit.norms.us_groupnorm import GroupNorm_
from models.normkit.norms.us_powernorm import PowerNorm
from models.normkit.norms.us_instancenorm import InsNorm, BatchNorm_
from models.normkit.norms.us_scalenorm import ScaleNorm
from models.normkit.norms.dtn import DTN

# DTN is for ViT models with constant resolution
# DTN_pool used pooling operation to reduce the high resolution

def NormSelect(norm_type, embed_dim, head_num=4, resolution=(14,14)):
    if norm_type == "layer":
        return LayerNorm(embed_dim)

    elif norm_type == "dtn":
        return DTN(embed_dim, group_num=head_num)

    elif norm_type == "dtn_var":
        return DTN(embed_dim, group_num=head_num, only_var=True)

    elif norm_type == "dtn_pool":
        return DTN(embed_dim, group_num=head_num, resolution=resolution)

    elif norm_type == "batch":
        return BatchNorm_(embed_dim)

    elif norm_type == 'power':
        return PowerNorm(embed_dim, group_num=head_num, warmup_iters=3000)

    elif norm_type == "insnorm":
        return InsNorm(embed_dim)

    elif norm_type == "gnorm":
        return GroupNorm_(num_groups=head_num, num_channels=embed_dim)

    elif norm_type == "scale":
        return ScaleNorm(embed_dim)
    
    else:
        NotImplementedError("norm: not implemented!")





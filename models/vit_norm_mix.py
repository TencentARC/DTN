import torch
import matplotlib
import math
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from models.normkit.norm_select import NormSelect


import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads,  mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_type='layer',  **kwargs):
        super().__init__()

        if norm_type=='layer':
            self.norm1 = nn.LayerNorm(dim)
        else:
            self.norm1 = NormSelect(norm_type, dim, num_heads)
        self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                            attn_drop=attn_drop, proj_drop=drop, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if norm_type=='layer':
            self.norm2 = nn.LayerNorm(dim)
        else:
            self.norm2 = NormSelect(norm_type, dim, num_heads)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                        act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x





class PatchEmbed(nn.Module):
    """ Image to Patch Embedding, from timm
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    




class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, tokens_type='transformer', pool = 'cls',
                    in_chans=3, num_classes=1000, embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_type='layer', norm_layer=nn.LayerNorm,
                 local_up_to_layer=10, use_pos_embed=True):
        super().__init__()
        #embed_dim *= num_heads
        print("The local-up-to-layer is set to {}".format(local_up_to_layer))
        self.num_classes = num_classes
        self.local_up_to_layer = local_up_to_layer
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.use_pos_embed = use_pos_embed


        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=dpr[i], norm_layer=norm_layer,
                norm_type=norm_type)
            if i<local_up_to_layer else
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=dpr[i], norm_layer=norm_layer,
                norm_type='layer')
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        
    
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        skip = {'pos_embed', 'cls_token'}
        for name, _ in self.named_parameters():
            if name.endswith('norm_weight'):
                skip.add(name)
        return skip

    @torch.jit.ignore
    def no_weight_decay2(self):
        skip = {'pos_embed', 'cls_token'}
        return skip

    def get_classifier(self):
        return self.head

    @torch.jit.ignore
    def get_gating_param(self):
        gating_params = set()
        for name, _ in self.named_parameters():
            if name.endswith('norm_weight'):
                gating_params.add(name)
        return gating_params

    @torch.jit.ignore
    def get_position_param(self):
        position_params = set()
        for name, _ in self.named_parameters():
            if name.endswith('pos_proj'):
                position_params.add(name)
        return position_params

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for u,blk in enumerate(self.blocks):
            if u == self.local_up_to_layer :
                x = torch.cat((cls_tokens, x), dim=1)
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x




@register_model
def vit_norm_ti_star(pretrained=False, norm_type='layer',  **kwargs):
    model = VisionTransformer(embed_dim=192, depth=12, num_heads=4, mlp_ratio=4.,
                            norm_type=norm_type,  **kwargs)
    return model

@register_model
def vit_norm_s(pretrained=False, norm_type='layer', **kwargs):

    model = VisionTransformer(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., norm_type=norm_type, **kwargs)
    return model


@register_model
def vit_norm_s_star(pretrained=False, norm_type='layer', **kwargs):

    model = VisionTransformer(embed_dim=432, depth=12, num_heads=9, mlp_ratio=4., norm_type=norm_type, **kwargs)
    return model


@register_model
def vit_norm_b(pretrained=False, norm_type='layer', **kwargs):
    model = VisionTransformer(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., 
                                norm_type=norm_type, **kwargs)
    return model

@register_model
def vit_norm_b_star(pretrained=False, norm_type='layer', **kwargs):
    model = VisionTransformer(embed_dim=768, depth=12, num_heads=16, mlp_ratio=4., 
                                norm_type=norm_type, **kwargs)
    return model


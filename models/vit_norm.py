import torch
import math
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.normkit.norm_select import NormSelect
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
import logging
_logger = logging.getLogger(__name__)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, heads, fn, norm_type='layer'):
        super().__init__()
        if norm_type=='layer':
          self.norm = nn.LayerNorm(dim)
        else:
          self.norm = NormSelect(norm_type, dim, head_num=heads)
        self.fn = fn
        self.norm_type = norm_type

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def get_attention_map(self, x, return_map = False):
        B, N, C, h= *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0)

        img_size = int(N**.5)
        ind = torch.arange(img_size).view(1,-1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size,img_size)
        indy = ind.repeat_interleave(img_size,dim=0).repeat_interleave(img_size,dim=1)
        indd = indx**2 + indy**2
        distances = indd**.5
        distances = distances.to('cuda')

        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        #dist /= N

        if return_map:
            return dist, attn_map
        else:
            return dist



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., drop_path=0.,norm_type='layer'):
        super().__init__()
        self.layers = nn.ModuleList([])
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, 
                        heads, 
                        Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                        norm_type=norm_type),
                PreNorm(dim, 
                        heads, 
                        FeedForward(dim, mlp_dim, dropout = dropout),
                        norm_type='layer'),
                DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity()
            ]))

    def forward(self, x):
        for attn, ff, drop_path in self.layers:
            x = drop_path(attn(x)) + x
            x = drop_path(ff(x)) + x
        return x




class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, tokens_type='transformer', 
                pool = 'cls', in_chans=3, 
                num_classes=1000, embed_dim=384, depth=12,
                num_heads=6, mlp_ratio=4., local_up_to_layer=10, 
                norm_type='layer', qkv_bias=False, qk_scale=None, 
                drop_rate=0., attn_drop_rate=0.,
                drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        image_size = img_size
        dim = embed_dim
        channels = in_chans
        heads = num_heads
        dim_head = dim // num_heads
        mlp_dim = int(dim * mlp_ratio)
        dropout = drop_rate
        emb_dropout = drop_rate

        self.mix_layers = local_up_to_layer
        _logger.info("The layers of mixnorm are {}".format(self.mix_layers))



        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)


        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, \
            'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
            p1 = patch_height, 
            p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if tokens_type == 'transformer':
            _logger.info("Using transformer")
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, 
                                            dropout=dropout, drop_path=drop_path_rate, 
                                            norm_type=norm_type)
        else:
            _logger.info("The underlying transformer cannot be found")



        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        trunc_normal_(self.cls_token, std=.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embedding.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embedding
        class_pos_embed = self.pos_embedding[:, 0]
        patch_pos_embed = self.pos_embedding[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.pos_embedding.patch_size
        h0 = h // self.pos_embedding.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


    def forward(self, img):
        _, _, w, h = img.shape
        x = self.to_patch_embedding(img)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)

        #x += self.pos_embedding[:, :(n + 1)]
        x = x + self.interpolate_pos_encoding(x,w,h)


        x = self.dropout(x)


        x = self.transformer(x)


        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]


        x = self.to_latent(x)
        return self.mlp_head(x)

@register_model
def vit_b(pretrained=False, norm_type='layer', **kwargs):  # adopt transformers for tokens to token

    model = ViT(tokens_type='transformer', embed_dim=768, depth=12, 
                num_heads=12, mlp_ratio=4., norm_type=norm_type, **kwargs)
    return model

@register_model
def vit_s(pretrained=False, norm_type='layer', **kwargs):  # adopt transformers for tokens to token

    model = ViT(tokens_type='transformer', embed_dim=384, depth=12, 
                num_heads=6, mlp_ratio=4., norm_type=norm_type, **kwargs)
    return model

@register_model
def vit_ti(pretrained=False, norm_type='layer', **kwargs):  # adopt transformers for tokens to token

    model = ViT(tokens_type='transformer', embed_dim=192, depth=12, 
                num_heads=3, mlp_ratio=4., norm_type=norm_type,**kwargs)
    return model

@register_model
def vit_ti_star(pretrained=False, norm_type='layer', **kwargs):  # adopt transformers for tokens to token

    model = ViT(tokens_type='transformer', embed_dim=192, depth=12, 
                num_heads=4, mlp_ratio=4., norm_type=norm_type, **kwargs)
    return model

@register_model
def vit_s_star(pretrained=False, norm_type='layer', **kwargs):  # adopt transformers for tokens to token

    model = ViT(tokens_type='transformer', embed_dim=432, depth=12, 
                num_heads=9, mlp_ratio=4., norm_type=norm_type, **kwargs)
    return model

@register_model
def vit_b_star(pretrained=False, norm_type='layer', **kwargs):  # adopt transformers for tokens to token

    model = ViT(tokens_type='transformer', embed_dim=768, depth=12, 
                num_heads=16, mlp_ratio=4., norm_type=norm_type, **kwargs)
    return model






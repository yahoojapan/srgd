import torch
from timm.utils import ModelEmaV2

from functools import partial
from collections import namedtuple

import math
import torch.nn as nn
import torch.nn.functional as F

from denoising_diffusion_pytorch import (
    Unet,
    GaussianDiffusion,
    ElucidatedDiffusion,
)
from denoising_diffusion_pytorch.attend import Attend

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.special import expm1
from torch import sqrt
from tqdm import tqdm
from torch.cuda.amp import autocast


# This code is significantly inspired by or directly copied from the
# "denoising-diffusion-pytorch" implementation found at the following GitHub repository:
# https://github.com/lucidrains/denoising-diffusion-pytorch
#
# All credit for the original implementation and concept goes to the authors
# of the "denoising-diffusion-pytorch" project. Any errors or shortcomings in
# this adaptation are my own.

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

## small helper modules

class PixelShuffleUpsample(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

def kldiv_loss(pred, target, reduction = 'none'):
    loss = torch.nn.KLDivLoss(reduction=reduction)(F.log_softmax(pred, dim=1), F.softmax(target, dim=1))
    return loss

def get_coord_and_pad(height, width, tile_size=256):
    if height <= tile_size and width <= tile_size:
        new_height, new_width = tile_size, tile_size
    else:
        new_height = ((height-1)//tile_size + 1) * tile_size + tile_size
        new_width = ((width-1)//tile_size + 1) * tile_size + tile_size

    left = (new_width - width) // 2
    top = (new_height - height) // 2
    right = left + width
    bottom = top + height
    coord = (left, top, right, bottom)

    pad_left = left
    pad_right = new_width - pad_left - width
    pad_top = top
    pad_bottom = new_height - pad_top - height
    pad = (pad_left, pad_right, pad_top, pad_bottom)

    return coord, pad

def get_coords(h, w, tile_size, tile_stride, diff=0):
    hi_list = list(range(0, h - tile_size + 1, tile_stride))
    if (h - tile_size) % tile_stride != 0:
        hi_list.append(h - tile_size)

    wi_list = list(range(0, w - tile_size + 1, tile_stride))
    if (w - tile_size) % tile_stride != 0:
        wi_list.append(w - tile_size)

    coords = []
    for hi in hi_list:
        for wi in wi_list:
            coords.append((hi + diff, hi + tile_size + diff, wi + diff, wi + tile_size + diff))
    return coords

def get_area(coords, height, width):
    top = height
    bottom = 0
    left = width
    right = 0

    for coord in coords:
        hs, he, ws, we = coord
        if hs < top:
            top = hs
        if he > bottom:
            bottom = he
        if ws < left:
            left = ws
        if we > right:
            right = we
    coord = (left, top, right, bottom)

    area_height = bottom - top
    area_width = right - left

    pad_left = left
    pad_right = width - pad_left - area_width
    pad_top = top
    pad_bottom = height - pad_top - area_height
    pad = (pad_left, pad_right, pad_top, pad_bottom)

    return coord, pad

import torch.fft as fft
def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).to(x.device)

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)



class SRUnet(Unet):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = True, # Set self_condition=True to allow input of LR images
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = (False, False, False, True),
        flash_attn = False,
        pixel_shuffle_upsample = True,
    ):
        super().__init__(
            dim,
            init_dim,
            out_dim,
            dim_mults,
            channels,
            self_condition,
            resnet_block_groups,
            learned_variance,
            learned_sinusoidal_cond,
            random_fourier_features,
            learned_sinusoidal_dim,
            attn_dim_head,
            attn_heads,
            full_attn,
            flash_attn
        )
        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        self.dims = dims
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        # upsample klass
        # Modify to enable the use of PixelshuffleUpsample
        upsample_klass = Upsample if not pixel_shuffle_upsample else PixelShuffleUpsample

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                upsample_klass(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)


    def forward(self, x, time, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            # x = torch.cat((x_self_cond, x), dim = 1)
            x = torch.cat((x, x_self_cond), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            h_ = h.pop()
            x = torch.cat((x, h_), dim = 1)
            x = block1(x, t)

            h_ = h.pop()
            x = torch.cat((x, h_), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


# SRUnet that can add class conditions
class ConditionalSRUnet(Unet):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = True, # Set self_condition=True to allow input of LR images
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = (False, False, False, True),
        flash_attn = False,
        pixel_shuffle_upsample = True,
        num_classes = None
    ):
        super().__init__(
            dim,
            init_dim,
            out_dim,
            dim_mults,
            channels,
            self_condition,
            resnet_block_groups,
            learned_variance,
            learned_sinusoidal_cond,
            random_fourier_features,
            learned_sinusoidal_dim,
            attn_dim_head,
            attn_heads,
            full_attn,
            flash_attn
        )
        # determine dimensions

        self.num_classes = num_classes

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # class conditional embeddings
        # Align with time_dim
        if self.num_classes is not None:
            class_emb = nn.Embedding(self.num_classes, dim)
            self.class_mlp = nn.Sequential(
                class_emb,
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )

        # attention

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        # upsample klass
        # Modify to enable the use of PixelshuffleUpsample
        upsample_klass = Upsample if not pixel_shuffle_upsample else PixelShuffleUpsample

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                upsample_klass(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)


    def forward(self, x, time, class_label = None, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            # x = torch.cat((x_self_cond, x), dim = 1)
            x = torch.cat((x, x_self_cond), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        # class conditional
        if class_label is not None:
            c = self.class_mlp(class_label)
            t = t + c

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# diffusion helpers

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusionSR(GaussianDiffusion):
    def set_seed(self, seed):
        torch.cuda.manual_seed(seed)

    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        cond_drop_prob = 0.,
        loss_type = 'l2',
    ):
        super().__init__(
            model=model,
            image_size=image_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            objective=objective,
            beta_schedule=beta_schedule,
            schedule_fn_kwargs=schedule_fn_kwargs,
            ddim_sampling_eta=ddim_sampling_eta,
            auto_normalize=auto_normalize,
            offset_noise_strength=offset_noise_strength,
            min_snr_loss_weight=min_snr_loss_weight,
            min_snr_gamma=min_snr_gamma
        )
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        self.cond_drop_prob = cond_drop_prob
        self.loss_type = loss_type

    def model_predictions(self, x, t, condition_x = None, cond_scale = 1.0, clip_x_start = False, rederive_pred_noise = False):
        if cond_scale == 1.0:
            model_output = self.model(x, t, condition_x)
        else:
            cond_out = self.model(x, t, condition_x)
            null_out = self.model(x, t, None)
            model_output = null_out + (cond_out - null_out) * cond_scale

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, condition_x = None, cond_scale = 1.0, clip_denoised = True):
        preds = self.model_predictions(x, t, condition_x, cond_scale)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, condition_x = None, cond_scale = 1.0):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, condition_x = condition_x, cond_scale = cond_scale, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, condition_x, cond_scale, guidance_start_steps, generation_start_steps, sampling_timesteps,
                      with_images, with_x0_images):
        batch, device = shape[0], self.device

        if generation_start_steps > 0:
            target_time = self.num_timesteps - generation_start_steps
            t = torch.tensor([target_time]*batch, device=device).long()
            img = self.q_sample(x_start=condition_x, t=t)
        else:
            img = torch.randn(shape, device = device)

        if with_images:
            image_list = []
            image_list.append(img.clone().detach().cpu())

        x_start = None

        if with_x0_images:
            x0_image_list = []
            x0_image_list.append(img.clone().detach().cpu())

        for i, t in enumerate(tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps)):
            if i < generation_start_steps:
                continue
            if i < guidance_start_steps:
                cur_cond_scale = 1.0
            else:
                cur_cond_scale = cond_scale
            img, x_start = self.p_sample(img, t, condition_x, cur_cond_scale)
            if with_images:
                image_list.append(img.clone().detach().cpu())
            if with_x0_images:
                x0_image_list.append(x_start.clone().detach().cpu())

        if with_images:
            if with_x0_images:
                return self.unnormalize(img), image_list, x0_image_list
            else:
                return self.unnormalize(img), image_list
        else:
            return self.unnormalize(img)

    @torch.inference_mode()
    def ddim_sample(self, shape, condition_x, cond_scale, guidance_start_steps, generation_start_steps, sampling_timesteps,
                    with_images, with_x0_images):
        batch, device, total_timesteps, eta = shape[0], self.device, self.num_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        if generation_start_steps > 0:
            target_time = time_pairs[generation_start_steps][0]
            t = torch.tensor([target_time]*batch, device=device).long()
            img = self.q_sample(x_start=condition_x, t=t)
        else:
            img = torch.randn(shape, device = device)

        if with_images:
            image_list = []
            image_list.append(img.clone().detach().cpu())

        x_start = None

        if with_x0_images:
            x0_image_list = []
            x0_image_list.append(img.clone().detach().cpu())

        for i, (time, time_next) in enumerate(tqdm(time_pairs, desc = 'sampling loop time step')):
            if i < generation_start_steps:
                continue
            if i < guidance_start_steps:
                cur_cond_scale = 1.0
            else:
                cur_cond_scale = cond_scale
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, condition_x, cur_cond_scale, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                if with_images:
                    image_list.append(img.clone().detach().cpu())
                if with_x0_images:
                    x0_image_list.append(img.clone().detach().cpu())
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if with_images:
                image_list.append(img.clone().detach().cpu())
            if with_x0_images:
                x0_image_list.append(x_start.clone().detach().cpu())

        if with_images:
            if with_x0_images:
                return self.unnormalize(img), image_list, x0_image_list
            else:
                return self.unnormalize(img), image_list
        else:
            return self.unnormalize(img)

    @torch.inference_mode()
    def tiled_sample(self, batch_size=4, tile_size=256, tile_stride=256,
                     condition_x=None, class_label=None,
                     cond_scale=1.0, guidance_start_steps=0,
                     class_cond_scale=1.0, class_guidance_start_steps=0,
                     generation_start_steps=0, num_sample_steps=None,
                     with_images=False, with_x0_images=False, start_white_noise=True, amp=False):

        num_sample_steps = default(num_sample_steps, self.sampling_timesteps)
        condition_x = normalize_to_neg_one_to_one(condition_x)

        batch, c, h, w = condition_x.shape

        # pad condition_x
        coord, pad = get_coord_and_pad(h, w)
        left, top, right, bottom = coord
        condition_x = F.pad(condition_x, pad, mode='reflect')

        device, total_timesteps, eta = self.device, self.num_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps = num_sample_steps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        if generation_start_steps > 0:
            target_time = time_pairs[generation_start_steps][0]
            t = torch.tensor([target_time]*batch, device=device).long()
            img = self.q_sample(x_start=condition_x, t=t)
        else:
            img = torch.randn(condition_x.shape, device = device)

        if with_images:
            image_list = []
            image_list.append(img.clone().detach().cpu())


        if with_x0_images:
            x0_image_list = []
            x0_image_list.append(img.clone().detach().cpu())

        # Pre-calculate tile regions
        _, _, height, width = condition_x.shape
        coords0 = get_coords(height, width, tile_size, tile_size, diff=0)
        if height <= tile_size and width <= tile_size:
            coords1 = get_coords(height, width, tile_size, tile_stride, diff=0)
        else:
            coords1 = get_coords(height-tile_size, width-tile_size, tile_size, tile_stride, diff=tile_size//2)
        coord_list = [coords0, coords1]

        # Get the region of the smaller coords
        small_coord, small_pad = get_area(coords1, height, width)
        sleft, stop, sright, sbottom = small_coord

        # Pad the outside of the smaller region of condition_x with 0
        cropped_condition_x = condition_x[:,:,stop:sbottom,sleft:sright]
        condition_x = F.pad(cropped_condition_x, small_pad, mode='constant', value=0)

        # x_start = None
        x_start = img.clone()
        pred_noise = torch.zeros_like(img)

        for i, (time, time_next) in enumerate(tqdm(time_pairs, desc = 'sampling loop time step')):
            if i < generation_start_steps:
                continue
            if i < guidance_start_steps:
                cur_cond_scale = 1.0
            else:
                cur_cond_scale = cond_scale

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

            cur_coords = coord_list[i%2]

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            cc = (1 - alpha_next - sigma ** 2).sqrt()

            minibatch_index = 0
            minibatch = torch.zeros((batch_size, c, tile_size, tile_size), device=condition_x.device)
            minibatch_condition = torch.zeros((batch_size, c, tile_size, tile_size), device=condition_x.device)
            output_indexes = [None] * batch_size
            for hs, he, ws, we in cur_coords:
                minibatch[minibatch_index] = img[:, :, hs:he, ws:we]
                minibatch_condition[minibatch_index] = condition_x[:, :, hs:he, ws:we]
                output_indexes[minibatch_index] = (hs, ws)
                minibatch_index += 1

                if minibatch_index == batch_size:
                    with autocast(enabled=amp):
                        tile_pred_noise, tile_x_start, *_ = self.model_predictions(minibatch, time_cond, minibatch_condition, cur_cond_scale, clip_x_start = True, rederive_pred_noise = True)

                    for k in range(minibatch_index):
                        hs, ws = output_indexes[k]
                        pred_noise[:, :, hs:hs+tile_size, ws:ws+tile_size] = tile_pred_noise[k]
                        x_start[:, :, hs:hs+tile_size, ws:ws+tile_size] = tile_x_start[k]
                        # cur_img = tile_x_start[k] * alpha_next.sqrt() + cc * tile_pred_noise[k] + sigma * noise[:, :, hs:hs+tile_size, ws:ws+tile_size]
                        # img[:, :, hs:hs+tile_size, ws:ws+tile_size] = cur_img

                    minibatch_index = 0

            if minibatch_index > 0:
                with autocast(enabled=amp):
                    tile_pred_noise, tile_x_start, *_ = self.model_predictions(minibatch[0:minibatch_index], time_cond, minibatch_condition[0:minibatch_index], cur_cond_scale, clip_x_start = True, rederive_pred_noise = True)

                for k in range(minibatch_index):
                    hs, ws = output_indexes[k]
                    pred_noise[:, :, hs:hs+tile_size, ws:ws+tile_size] = tile_pred_noise[k]
                    x_start[:, :, hs:hs+tile_size, ws:ws+tile_size] = tile_x_start[k]
                    # cur_img = tile_x_start[k] * alpha_next.sqrt() + cc * tile_pred_noise[k] + sigma * noise[:, :, hs:hs+tile_size, ws:ws+tile_size]
                    # img[:, :, hs:hs+tile_size, ws:ws+tile_size] = cur_img

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  cc * pred_noise + \
                  sigma * noise

            if time_next < 0:
                img = x_start
                if with_images:
                    image_list.append(img.clone().detach().cpu())
                if with_x0_images:
                    x0_image_list.append(img.clone().detach().cpu())
                continue

            if i%2 == 1:
                # Reconstruct by removing the padding part of img when odd times
                cropped_img = img[:,:,stop:sbottom,sleft:sright]
                t = torch.tensor([time_next]*batch, device=device).long()
                img = self.q_sample(x_start=torch.zeros_like(condition_x), t=t)
                img[:,:,stop:sbottom,sleft:sright] = cropped_img

            if with_images:
                image_list.append(img.clone().detach().cpu())
            if with_x0_images:
                x0_image_list.append(x_start.clone().detach().cpu())

        img = img[:,:,top:bottom,left:right]
        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)

        if with_images:
            if with_x0_images:
                return img, image_list, x0_image_list
            else:
                return img, image_list
        else:
            return img

    @torch.inference_mode()
    def sample(self, batch_size = 16, condition_x = None, cond_scale = 1.0,
               guidance_start_steps = 0, generation_start_steps = 0,
               num_sample_steps = None, with_images = False, with_x0_images = False):
        # image_size, channels = self.image_size, self.channels
        sampling_timesteps = default(num_sample_steps, self.sampling_timesteps)

        _n, _c, h, w = condition_x.shape
        condition_x = normalize_to_neg_one_to_one(condition_x)
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, self.channels, h, w), condition_x, cond_scale,
                          guidance_start_steps, generation_start_steps, sampling_timesteps,
                          with_images, with_x0_images)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, condition_x,
                 noise = None, offset_noise_strength = None, clip_x_start = True):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        x_self_cond = condition_x

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        current_loss_weight = extract(self.loss_weight, t, loss.shape)
        loss = loss * current_loss_weight
        loss = reduce(loss, 'b ... -> b', 'mean')

        return loss.mean()

    def forward(self, img, condition_x, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        if torch.rand(1) < self.cond_drop_prob:
            condition_x = None
        else:
            condition_x = self.normalize(condition_x)

        return self.p_losses(img, t, condition_x, *args, **kwargs)



class ConditionalGaussianDiffusionSR(GaussianDiffusion):
    def set_seed(self, seed):
        torch.cuda.manual_seed(seed)

    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        cond_drop_prob = 0.,
        class_cond_drop_prob = 0.,
        loss_type = 'l2',
    ):
        super().__init__(
            model=model,
            image_size=image_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            objective=objective,
            beta_schedule=beta_schedule,
            schedule_fn_kwargs=schedule_fn_kwargs,
            ddim_sampling_eta=ddim_sampling_eta,
            auto_normalize=auto_normalize,
            offset_noise_strength=offset_noise_strength,
            min_snr_loss_weight=min_snr_loss_weight,
            min_snr_gamma=min_snr_gamma
        )
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        self.cond_drop_prob = cond_drop_prob
        self.class_cond_drop_prob = class_cond_drop_prob
        self.loss_type = loss_type

    def model_predictions(self, x, t, condition_x = None, class_label = None,
                          cond_scale = 1.0, class_cond_scale = 1.0,
                          clip_x_start = False, rederive_pred_noise = False):

        # Currently not supported for CFG with both condition_x and class_label
        if (cond_scale != 1.0) and (class_cond_scale != 1.0):
            raise NotImplementedError("Currently, you cannot specify both cond_scale and class_cond_scale at the same time.")

        if cond_scale == 1.0 and class_cond_scale == 1.0:
            model_output = self.model(x, t, class_label, condition_x)
        elif cond_scale != 1.0:
            cond_out = self.model(x, t, class_label, condition_x)
            null_out = self.model(x, t, class_label, None)
            model_output = null_out + (cond_out - null_out) * cond_scale
        elif class_cond_scale != 1.0:
            cond_out = self.model(x, t, class_label, condition_x)
            null_out = self.model(x, t, None, condition_x)
            model_output = null_out + (cond_out - null_out) * class_cond_scale

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, condition_x = None, class_label = None,
                        cond_scale = 1.0, class_cond_scale = 1.0, clip_denoised = True):
        preds = self.model_predictions(x, t, condition_x, class_label, cond_scale, class_cond_scale)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, condition_x = None, class_label = None,
                 cond_scale = 1.0, class_cond_scale = 1.0):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times,
                                                                          condition_x = condition_x, class_label = class_label,
                                                                          cond_scale = cond_scale,
                                                                          class_cond_scale = class_cond_scale,
                                                                          clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, condition_x, class_label,
                      cond_scale, guidance_start_steps,
                      class_cond_scale, class_guidance_start_steps,
                      generation_start_steps, sampling_timesteps, with_images, with_x0_images):
        batch, device = shape[0], self.device

        if generation_start_steps > 0:
            target_time = self.num_timesteps - generation_start_steps
            t = torch.tensor([target_time]*batch, device=device).long()
            img = self.q_sample(x_start=condition_x, t=t)
        else:
            img = torch.randn(shape, device = device)

        if with_images:
            image_list = []
            image_list.append(img.clone().detach().cpu())

        x_start = None

        if with_x0_images:
            x0_image_list = []
            x0_image_list.append(img.clne().detach().cpu())

        for i, t in enumerate(tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps)):
            if i < generation_start_steps:
                continue
            if i < guidance_start_steps:
                cur_cond_scale = 1.0
            else:
                cur_cond_scale = cond_scale
            if i < class_guidance_start_steps:
                cur_class_cond_scale = 1.0
            else:
                cur_class_cond_scale = class_cond_scale
            img, x_start = self.p_sample(img, t, condition_x, class_label, cur_cond_scale, cur_class_cond_scale)
            if with_images:
                image_list.append(img.clone().detach().cpu())
            if with_x0_images:
                x0_image_list.append(x_start.clone().detach().cpu())

        if with_images:
            if with_x0_images:
                return self.unnormalize(img), image_list, x0_image_list
            else:
                return self.unnormalize(img), image_list
        else:
            return self.unnormalize(img)

    @torch.inference_mode()
    def ddim_sample(self, shape, condition_x, class_label,
                    cond_scale, guidance_start_steps,
                    class_cond_scale, class_guidance_start_steps,
                    generation_start_steps, sampling_timesteps, with_images, with_x0_images):
        # batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        batch, device, total_timesteps, eta = shape[0], self.device, self.num_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        if generation_start_steps > 0:
            target_time = time_pairs[generation_start_steps][0]
            t = torch.tensor([target_time]*batch, device=device).long()
            img = self.q_sample(x_start=condition_x, t=t)
        else:
            img = torch.randn(shape, device = device)

        if with_images:
            image_list = []
            image_list.append(img.clone().detach().cpu())

        x_start = None

        if with_x0_images:
            x0_image_list = []
            x0_image_list.append(img.clone().detach().cpu())

        for i, (time, time_next) in enumerate(tqdm(time_pairs, desc = 'sampling loop time step')):
            if i < generation_start_steps:
                continue
            if i < guidance_start_steps:
                cur_cond_scale = 1.0
            else:
                cur_cond_scale = cond_scale
            if i < class_guidance_start_steps:
                cur_class_cond_scale = 1.0
            else:
                cur_class_cond_scale = class_cond_scale
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, condition_x, class_label,
                                                             cur_cond_scale, cur_class_cond_scale,
                                                             clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                if with_images:
                    image_list.append(img.clone().detach().cpu())
                if with_x0_images:
                    x0_image_list.append(img.clone().detach().cpu())
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if with_images:
                image_list.append(img.clone().detach().cpu())
            if with_x0_images:
                x0_image_list.append(x_start.clone().detach().cpu())

        if with_images:
            if with_x0_images:
                return self.unnormalize(img), image_list, x0_image_list
            else:
                return self.unnormalize(img), image_list
        else:
            return self.unnormalize(img)


    @torch.inference_mode()
    def sample(self, batch_size = 16, condition_x = None, class_label = None,
               cond_scale = 1.0, guidance_start_steps = 0,
               class_cond_scale = 1.0, class_guidance_start_steps = 0,
               generation_start_steps = 0,
               num_sample_steps = None, with_images = False, with_x0_images = False):
        # image_size, channels = self.image_size, self.channels
        sampling_timesteps = default(num_sample_steps, self.sampling_timesteps)

        _n, _c, h, w = condition_x.shape
        condition_x = normalize_to_neg_one_to_one(condition_x)
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, self.channels, h, w), condition_x, class_label,
                         cond_scale, guidance_start_steps,
                         class_cond_scale, class_guidance_start_steps,
                         generation_start_steps, sampling_timesteps, with_images, with_x0_images)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, class_label, condition_x,
                 noise = None, offset_noise_strength = None, clip_x_start=True):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        x_self_cond = condition_x

        # predict and take gradient step

        model_out = self.model(x, t, class_label, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        current_loss_weight = extract(self.loss_weight, t, loss.shape)
        loss = loss * current_loss_weight
        loss = reduce(loss, 'b ... -> b', 'mean')

        return loss.mean()

    def forward(self, img, condition_x, class_label, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        if torch.rand(1) < self.cond_drop_prob:
            condition_x = None
        else:
            condition_x = self.normalize(condition_x)

        if torch.rand(1) < self.class_cond_drop_prob:
            class_label = None

        return self.p_losses(img, t, class_label, condition_x, *args, **kwargs)


class ElucidatedDiffusionSR(ElucidatedDiffusion):
    def set_seed(self, seed):
        torch.cuda.manual_seed(seed)

    def __init__(
        self,
        net,
        *,
        image_size,
        channels = 3,
        num_sample_steps = 32, # number of sampling steps
        sigma_min = 0.002,     # min noise level
        sigma_max = 80,        # max noise level
        sigma_data = 0.5,      # standard deviation of data distribution
        rho = 7,               # controls the sampling schedule
        P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
        cond_drop_prob = 0.,
        use_dpmpp_solver = False,
        loss_type = 'l2'
    ):
        super().__init__(
            net=net,
            image_size=image_size,
            channels=channels,
            num_sample_steps=num_sample_steps,
            sigma_min=sigma_min, sigma_max=sigma_max, sigma_data=sigma_data,
            rho=rho,
            P_mean=P_mean, P_std=P_std,
            S_churn=S_churn, S_tmin=S_tmin, S_tmax=S_tmax, S_noise=S_noise
        )
        #assert net.learned_sinusoidal_cond
        assert net.random_or_learned_sinusoidal_cond
        self.self_condition = net.self_condition

        self.net = net

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # parameters

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

        self.cond_drop_prob = cond_drop_prob
        self.use_dpmpp_solver = use_dpmpp_solver
        self.loss_type = loss_type

    def set_seed(self, seed):
        torch.cuda.manual_seed(seed)

    def preconditioned_network_forward(self, noised_images, sigma, condition_x, cond_scale = 1.0, clamp = False):
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device = device)

        padded_sigma = rearrange(sigma, 'b -> b 1 1 1')

        net_out = self.net(
            self.c_in(padded_sigma) * noised_images,
            self.c_noise(sigma),
            condition_x
        )

        out = self.c_skip(padded_sigma) * noised_images +  self.c_out(padded_sigma) * net_out

        if cond_scale != 1.0:
            null_out = self.net(
                self.c_in(padded_sigma) * noised_images,
                self.c_noise(sigma),
                None
            )

            null_out = self.c_skip(padded_sigma) * noised_images + self.c_out(padded_sigma) * null_out

            out =  null_out + (out - null_out) * cond_scale

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    @torch.inference_mode()
    def get_noised_images(self, condition_x, target_step, num_sample_steps=None):
        # Input condition_x that has been normalize_to_neg_one_to_one
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        sigmas = self.sample_schedule(num_sample_steps)
        n, _, _, _ = condition_x.shape
        padded_sigmas = repeat(sigmas[target_step], ' -> b', b = n)
        noise = torch.randn_like(condition_x)
        noised_images = condition_x + padded_sigmas * noise  # alphas are 1. in the paper
        return noised_images

    @torch.inference_mode()
    def sample(self, batch_size=16, condition_x=None, cond_scale=1.0, guidance_start_steps=0,
               generation_start_steps=0,
               num_sample_steps=None, clamp=True, with_images=False, with_x0_images=False, zero_init=False):
        if self.use_dpmpp_solver:
            return self.sample_using_dpmpp(batch_size, condition_x, cond_scale, guidance_start_steps,
                                           generation_start_steps, num_sample_steps, clamp, with_images, with_x0_images, zero_init)
        else:
            return self.sample_org(batch_size, condition_x, cond_scale, guidance_start_steps,
                                   generation_start_steps, num_sample_steps, clamp, with_images, with_x0_images, zero_init)

    @torch.inference_mode()
    def sample_org(self, batch_size = 16, condition_x = None, cond_scale = 1.0, guidance_start_steps = 0,
                   generation_start_steps = 0, num_sample_steps = None, clamp = True,
                   with_images = False, with_x0_images = False, zero_init = False):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        _n, _c, h, w = condition_x.shape
        # image_size = h
        shape = (batch_size, self.channels, h, w)

        condition_x = normalize_to_neg_one_to_one(condition_x)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, math.sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        if generation_start_steps > 0:
            images = self.get_noised_images(condition_x, generation_start_steps)
        elif zero_init:
            images = torch.zeros(shape, device = self.device)
        else:
            # images is noise at the beginning
            init_sigma = sigmas[0]
            images = init_sigma * torch.randn(shape, device = self.device)

        if with_images:
            image_list = []
            image_list.append(images.clone().detach().cpu())

        if with_x0_images:
            x0_image_list = []
            x0_image_list.append(images.clone().detach().cpu())

        # gradually denoise

        for i, (sigma, sigma_next, gamma) in enumerate(tqdm(sigmas_and_gammas, desc = 'sampling time step')):
            if i < generation_start_steps:
                continue
            if i < guidance_start_steps:
                cur_cond_scale = 1.0
            else:
                cur_cond_scale = cond_scale
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device = self.device) # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            images_hat = images + math.sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            model_output = self.preconditioned_network_forward(images_hat, sigma_hat, condition_x, cur_cond_scale, clamp = clamp)
            denoised_over_sigma = (images_hat - model_output) / sigma_hat

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(images_next, sigma_next, condition_x, cur_cond_scale, clamp = clamp)
                denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

            images = images_next

            if with_images:
                image_list.append(images.clone().detach().cpu())
            if with_x0_images:
                if sigma_next != 0:
                    x0_image_list.append(denoised_prime_over_sigma.clone().detach().cpu())
                else:
                    x0_image_list.append(denoised_over_sigma.clone().detach().cpu())

        images = images.clamp(-1., 1.)

        if with_images:
            if with_x0_images:
                return unnormalize_to_zero_to_one(images), image_list, x0_image_list
            else:
                return unnormalize_to_zero_to_one(images), image_list
        else:
            return unnormalize_to_zero_to_one(images)

    @torch.inference_mode()
    def sample_using_dpmpp(self, batch_size = 16, condition_x = None, cond_scale = 1.0, guidance_start_steps = 0,
                           generation_start_steps = 0, num_sample_steps = None, clamp = True,
                           with_images = False, with_x0_images = False, zero_init = False):
        """
        thanks to Katherine Crowson (https://github.com/crowsonkb) for figuring it all out!
        https://arxiv.org/abs/2211.01095
        """
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        _n, _c, h, w = condition_x.shape
        # image_size = h
        shape = (batch_size, self.channels, h, w)

        condition_x = normalize_to_neg_one_to_one(condition_x)

        sigmas = self.sample_schedule(num_sample_steps)

        if generation_start_steps > 0:
            images = self.get_noised_images(condition_x, generation_start_steps)
        elif zero_init:
            images = torch.zeros(shape, device = self.device)
        else:
            images  = sigmas[0] * torch.randn(shape, device = self.device)

        if with_images:
            image_list = []
            image_list.append(images.clone().detach().cpu())

        if with_x0_images:
            x0_image_list = []
            x0_image_list.append(images.clone().detach().cpu())

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        old_denoised = None
        for i in tqdm(range(len(sigmas) - 1)):
            if i < generation_start_steps:
                continue
            if i < guidance_start_steps:
                cur_cond_scale = 1.0
            else:
                cur_cond_scale = cond_scale
            denoised = self.preconditioned_network_forward(images, sigmas[i].item(), condition_x, cur_cond_scale, clamp = clamp)
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t

            if not exists(old_denoised) or sigmas[i + 1] == 0:
                denoised_d = denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                gamma = - 1 / (2 * r)
                denoised_d = (1 - gamma) * denoised + gamma * old_denoised

            images = (sigma_fn(t_next) / sigma_fn(t)) * images - (-h).expm1() * denoised_d
            old_denoised = denoised

            if with_images:
                image_list.append(images.clone().detach().cpu())
            if with_x0_images:
                x0_image_list.append(denoised_d.clone().detach().cpu())

        images = images.clamp(-1., 1.)

        if with_images:
            if with_x0_images:
                return unnormalize_to_zero_to_one(images), image_list, x0_image_list
            else:
                return unnormalize_to_zero_to_one(images), image_list
        else:
            return unnormalize_to_zero_to_one(images)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def forward(self, images, condition_x):
        batch_size, c, h, w, device, image_size, channels = *images.shape, images.device, self.image_size, self.channels

        assert h == image_size and w == image_size, f'height and width of image must be {image_size}'
        assert c == channels, 'mismatch of image channels'

        images = normalize_to_neg_one_to_one(images)
        if torch.randn(1) < self.cond_drop_prob:
            condition_x = None
        else:
            condition_x = normalize_to_neg_one_to_one(condition_x)

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1')

        noise = torch.randn_like(images)

        noised_images = images + padded_sigmas * noise  # alphas are 1. in the paper

        denoised = self.preconditioned_network_forward(noised_images, sigmas, condition_x, cond_scale=1.0)

        losses = self.loss_fn(denoised, images, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        losses = losses * self.loss_weight(sigmas)

        return losses.mean()


class ConditionalElucidatedDiffusionSR(ElucidatedDiffusion):
    def set_seed(self, seed):
        torch.cuda.manual_seed(seed)

    def __init__(
        self,
        net,
        *,
        image_size,
        channels = 3,
        num_sample_steps = 32, # number of sampling steps
        sigma_min = 0.002,     # min noise level
        sigma_max = 80,        # max noise level
        sigma_data = 0.5,      # standard deviation of data distribution
        rho = 7,               # controls the sampling schedule
        P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
        cond_drop_prob = 0.,
        class_cond_drop_prob = 0.,
        use_dpmpp_solver = False,
        loss_type = 'l2'
    ):
        super().__init__(
            net=net,
            image_size=image_size,
            channels=channels,
            num_sample_steps=num_sample_steps,
            sigma_min=sigma_min, sigma_max=sigma_max, sigma_data=sigma_data,
            rho=rho,
            P_mean=P_mean, P_std=P_std,
            S_churn=S_churn, S_tmin=S_tmin, S_tmax=S_tmax, S_noise=S_noise
        )
        #assert net.learned_sinusoidal_cond
        assert net.random_or_learned_sinusoidal_cond
        self.self_condition = net.self_condition

        self.net = net

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # parameters

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

        self.cond_drop_prob = cond_drop_prob
        self.class_cond_drop_prob = class_cond_drop_prob
        self.use_dpmpp_solver = use_dpmpp_solver
        self.loss_type = loss_type

    def set_seed(self, seed):
        torch.cuda.manual_seed(seed)

    def preconditioned_network_forward(self, noised_images, sigma, condition_x, class_label,
                                       cond_scale = 1.0, class_cond_scale = 1.0, clamp = False):
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device = device)

        padded_sigma = rearrange(sigma, 'b -> b 1 1 1')

        net_out = self.net(
            self.c_in(padded_sigma) * noised_images,
            self.c_noise(sigma),
            class_label,
            condition_x
        )

        out = self.c_skip(padded_sigma) * noised_images +  self.c_out(padded_sigma) * net_out

        # Currently not supported for CFG with both condition_x and class_label
        if (cond_scale != 1.0) and (class_cond_scale != 1.0):
            raise NotImplementedError("Currently, you cannot specify both cond_scale and class_cond_scale at the same time.")

        # CFG by condition_x
        if cond_scale != 1.0:
            null_out = self.net(
                self.c_in(padded_sigma) * noised_images,
                self.c_noise(sigma),
                class_label,
                None
            )

            null_out = self.c_skip(padded_sigma) * noised_images + self.c_out(padded_sigma) * null_out

            out =  null_out + (out - null_out) * cond_scale

        # CFG by class_label
        if class_cond_scale != 1.0:
            null_out = self.net(
                self.c_in(padded_sigma) * noised_images,
                self.c_noise(sigma),
                None,
                condition_x
            )

            null_out = self.c_skip(padded_sigma) * noised_images + self.c_out(padded_sigma) * null_out

            out =  null_out + (out - null_out) * class_cond_scale

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    @torch.inference_mode()
    def get_noised_images(self, condition_x, target_step, num_sample_steps=None):
        # Input condition_x that has been normalize_to_neg_one_to_one
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        sigmas = self.sample_schedule(num_sample_steps)
        n, _, _, _ = condition_x.shape
        padded_sigmas = repeat(sigmas[target_step], ' -> b', b = n)
        noise = torch.randn_like(condition_x)
        noised_images = condition_x + padded_sigmas * noise  # alphas are 1. in the paper
        return noised_images

    @torch.inference_mode()
    def sample(self, batch_size=16, condition_x=None, class_label=None, cond_scale=1.0, guidance_start_steps=0,
               class_cond_scale=1.0, class_guidance_start_steps=0, generation_start_steps=0,
               num_sample_steps=None, clamp=True, with_images=False, with_x0_images=False, zero_init=False):
        if self.use_dpmpp_solver:
            return self.sample_using_dpmpp(batch_size, condition_x, class_label,
                                           cond_scale, guidance_start_steps,
                                           class_cond_scale, class_guidance_start_steps,
                                           generation_start_steps, num_sample_steps, clamp, with_images, with_x0_images, zero_init)
        else:
            return self.sample_org(batch_size, condition_x, class_label,
                                   cond_scale, guidance_start_steps,
                                   class_cond_scale, class_guidance_start_steps,
                                   generation_start_steps, num_sample_steps, clamp, with_images, with_x0_images, zero_init)

    @torch.inference_mode()
    def sample_org(self, batch_size = 16, condition_x = None, class_label = None,
                   cond_scale = 1.0, guidance_start_steps = 0,
                   class_cond_scale = 1.0, class_guidance_start_steps = 0,
                   generation_start_steps = 0, num_sample_steps = None, clamp = True,
                   with_images = False, with_x0_images = False, zero_init = False):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        _n, _c, h, w = condition_x.shape
        # image_size = h
        shape = (batch_size, self.channels, h, w)

        condition_x = normalize_to_neg_one_to_one(condition_x)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, math.sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        if generation_start_steps > 0:
            images = self.get_noised_images(condition_x, generation_start_steps)
        elif zero_init:
            images = torch.zeros(shape, device = self.device)
        else:
            # images is noise at the beginning
            init_sigma = sigmas[0]
            images = init_sigma * torch.randn(shape, device = self.device)

        if with_images:
            image_list = []
            image_list.append(images.clone().detach().cpu())

        if with_x0_images:
            x0_image_list = []
            x0_image_list.append(images.clone().detach().cpu())

        # gradually denoise

        for i, (sigma, sigma_next, gamma) in enumerate(tqdm(sigmas_and_gammas, desc = 'sampling time step')):
            if i < generation_start_steps:
                continue
            if i < guidance_start_steps:
                cur_cond_scale = 1.0
            else:
                cur_cond_scale = cond_scale
            if i < class_guidance_start_steps:
                cur_class_cond_scale = 1.0
            else:
                cur_class_cond_scale = class_cond_scale
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device = self.device) # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            images_hat = images + math.sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            model_output = self.preconditioned_network_forward(images_hat, sigma_hat, condition_x, class_label,
                                                               cur_cond_scale, cur_class_cond_scale, clamp = clamp)
            denoised_over_sigma = (images_hat - model_output) / sigma_hat

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(images_next, sigma_next, condition_x, class_label,
                                                                        cur_cond_scale, cur_class_cond_scale, clamp = clamp)
                denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

            images = images_next

            if with_images:
                image_list.append(images.clone().detach().cpu())
            if with_x0_images:
                if sigma_next != 0:
                    x0_image_list.append(denoised_prime_over_sigma.clone().detach().cpu())
                else:
                    x0_image_list.append(denoised_over_sigma.clone().detach().cpu())

        images = images.clamp(-1., 1.)

        if with_images:
            if with_x0_images:
                return unnormalize_to_zero_to_one(images), image_list, x0_image_list
            else:
                return unnormalize_to_zero_to_one(images), image_list
        else:
            return unnormalize_to_zero_to_one(images)

    @torch.inference_mode()
    def tiled_sample(self, batch_size=4, tile_size=256, tile_stride=256,
                         condition_x=None, class_label=None,
                         cond_scale=1.0, guidance_start_steps=0,
                         class_cond_scale=1.0, class_guidance_start_steps=0,
                         generation_start_steps=0, num_sample_steps=None, clamp = True, zero_init = False,
                         with_images=False, with_x0_images=False, start_white_noise=True, amp=False):

        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        condition_x = normalize_to_neg_one_to_one(condition_x)

        batch, c, h, w = condition_x.shape

        # pad condition_x
        coord, pad = get_coord_and_pad(h, w)
        left, top, right, bottom = coord
        condition_x = F.pad(condition_x, pad, mode='reflect')

        # shape = (batch_size, self.channels, h, w)
        shape = condition_x.shape

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, math.sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        if generation_start_steps > 0:
            images = self.get_noised_images(condition_x, generation_start_steps)
        elif zero_init:
            images = torch.zeros(shape, device = self.device)
        else:
            # images is noise at the beginning
            init_sigma = sigmas[0]
            images = init_sigma * torch.randn(shape, device = self.device)

        if with_images:
            image_list = []
            image_list.append(images[:,:,top:bottom,left:right].clone().detach().cpu())

        if with_x0_images:
            x0_image_list = []
            x0_image_list.append(images[:,:,top:bottom,left:right].clone().detach().cpu())

        # Pre-calculate tile regions
        _, _, height, width = condition_x.shape
        coords0 = get_coords(height, width, tile_size, tile_size, diff=0)
        if height <= tile_size and width <= tile_size:
            coords1 = get_coords(height, width, tile_size, tile_stride, diff=0)
        else:
            coords1 = get_coords(height-tile_size, width-tile_size, tile_size, tile_stride, diff=tile_size//2)
        coord_list = [coords0, coords1]

        # Get the region of the smaller coords
        small_coord, small_pad = get_area(coords1, height, width)
        sleft, stop, sright, sbottom = small_coord

        # Pad the outside of the smaller region of condition_x with 0
        cropped_condition_x = condition_x[:,:,stop:sbottom,sleft:sright]
        condition_x = F.pad(cropped_condition_x, small_pad, mode='constant', value=0)

        # gradually denoise

        x_start = images.clone()

        for i, (sigma, sigma_next, gamma) in enumerate(tqdm(sigmas_and_gammas, desc = 'sampling time step')):
            if i < generation_start_steps:
                continue
            if i < guidance_start_steps:
                cur_cond_scale = 1.0
            else:
                cur_cond_scale = cond_scale
            if i < class_guidance_start_steps:
                cur_class_cond_scale = 1.0
            else:
                cur_class_cond_scale = class_cond_scale
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device = self.device) # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            images_hat = images + math.sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            cur_coords = coord_list[i%2]

            minibatch_index = 0
            minibatch = torch.zeros((batch_size, c, tile_size, tile_size), device=condition_x.device)
            minibatch_condition = torch.zeros((batch_size, c, tile_size, tile_size), device=condition_x.device)
            output_indexes = [None] * batch_size
            for hs, he, ws, we in cur_coords:
                minibatch[minibatch_index] = images_hat[:, :, hs:he, ws:we]
                minibatch_condition[minibatch_index] = condition_x[:, :, hs:he, ws:we]
                output_indexes[minibatch_index] = (hs, ws)
                minibatch_index += 1

                if minibatch_index == batch_size:
                    with autocast(enabled=amp):
                        tile_out = self.preconditioned_network_forward(minibatch, sigma_hat, minibatch_condition, class_label,
                                                                       cur_cond_scale, cur_class_cond_scale, clamp=clamp)
                        denoised_over_sigma = (minibatch - tile_out) / sigma_hat
                        images_next = minibatch + (sigma_next - sigma_hat) * denoised_over_sigma
                    # second order correction, if not the last timestep
                    if sigma_next != 0:
                        tile_out_next = self.preconditioned_network_forward(images_next, sigma_next, minibatch_condition, class_label,
                                                                            cur_cond_scale, cur_class_cond_scale, clamp=clamp)
                        denoised_prime_over_sigma = (images_next - tile_out_next) / sigma_next
                        images_next = minibatch + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

                    for k in range(minibatch_index):
                        hs, ws = output_indexes[k]
                        images[:,:, hs:hs+tile_size, ws:ws+tile_size] = images_next[k]
                        if sigma_next != 0:
                            x_start[:, :, hs:hs+tile_size, ws:ws+tile_size] = denoised_prime_over_sigma[k]
                        else:
                            x_start[:, :, hs:hs+tile_size, ws:ws+tile_size] = denoised_over_sigma[k]
                    minibatch_index = 0

            if minibatch_index > 0:
                with autocast(enabled=amp):
                    tile_out = self.preconditioned_network_forward(minibatch[0:minibatch_index], sigma_hat,
                                                                   minibatch_condition[0:minibatch_index], class_label,
                                                                   cur_cond_scale, cur_class_cond_scale, clamp=clamp)
                    denoised_over_sigma = (minibatch[0:minibatch_index] - tile_out) / sigma_hat
                    images_next = minibatch[0:minibatch_index] + (sigma_next - sigma_hat) * denoised_over_sigma
                # second order correction, if not the last timestep
                if sigma_next != 0:
                    tile_out_next = self.preconditioned_network_forward(images_next, sigma_next,
                                                                        minibatch_condition[0:minibatch_index], class_label,
                                                                        cur_cond_scale, cur_class_cond_scale, clamp=clamp)
                    denoised_prime_over_sigma = (images_next - tile_out_next) / sigma_next
                    images_next = minibatch[0:minibatch_index] + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

                for k in range(minibatch_index):
                    hs, ws = output_indexes[k]
                    images[:,:, hs:hs+tile_size, ws:ws+tile_size] = images_next[k]
                    if sigma_next != 0:
                        x_start[:, :, hs:hs+tile_size, ws:ws+tile_size] = denoised_prime_over_sigma[k]
                    else:
                        x_start[:, :, hs:hs+tile_size, ws:ws+tile_size] = denoised_over_sigma[k]

            if i%2 == 1:
                # Reconstruct by removing the padding part of img when odd times
                cropped_img = images[:,:,stop:sbottom,sleft:sright]
                images = self.get_noised_images(torch.zeros_like(condition_x), i)
                images[:,:,stop:sbottom,sleft:sright] = cropped_img

            if with_images:
                image_list.append(images.clone().detach().cpu())
            if with_x0_images:
                x0_image_list.append(x_start.clone().detach().cpu())

        images = images[:,:,top:bottom,left:right]
        images = images.clamp(-1., 1.)
        images = unnormalize_to_zero_to_one(images)

        if with_images:
            if with_x0_images:
                return images, image_list, x0_image_list
            else:
                return images, image_list
        else:
            return images


    @torch.inference_mode()
    def sample_using_dpmpp(self, batch_size = 16, condition_x = None, class_label = None,
                           cond_scale = 1.0, guidance_start_steps = 0,
                           class_cond_scale = 1.0, class_guidance_start_steps = 0,
                           generation_start_steps = 0, num_sample_steps = None, clamp = True,
                           with_images = False, with_x0_images = False, zero_init = False):
        """
        thanks to Katherine Crowson (https://github.com/crowsonkb) for figuring it all out!
        https://arxiv.org/abs/2211.01095
        """
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        _n, _c, h, w = condition_x.shape
        # image_size = h
        shape = (batch_size, self.channels, h, w)

        condition_x = normalize_to_neg_one_to_one(condition_x)

        sigmas = self.sample_schedule(num_sample_steps)

        if generation_start_steps > 0:
            images = self.get_noised_images(condition_x, generation_start_steps)
        elif zero_init:
            images = torch.zeros(shape, device = self.device)
        else:
            images  = sigmas[0] * torch.randn(shape, device = self.device)

        if with_images:
            image_list = []
            image_list.append(images.clone().detach().cpu())

        if with_x0_images:
            x0_image_list = []
            x0_image_list.append(images.clone().detach().cpu())

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        old_denoised = None
        for i in tqdm(range(len(sigmas) - 1)):
            if i < generation_start_steps:
                continue
            if i < guidance_start_steps:
                cur_cond_scale = 1.0
            else:
                cur_cond_scale = cond_scale
            if i < class_guidance_start_steps:
                cur_class_cond_scale = 1.0
            else:
                cur_class_cond_scale = class_cond_scale
            denoised = self.preconditioned_network_forward(images, sigmas[i].item(), condition_x, class_label,
                                                           cur_cond_scale, cur_class_cond_scale, clamp = clamp)
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t

            if not exists(old_denoised) or sigmas[i + 1] == 0:
                denoised_d = denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                gamma = - 1 / (2 * r)
                denoised_d = (1 - gamma) * denoised + gamma * old_denoised

            images = (sigma_fn(t_next) / sigma_fn(t)) * images - (-h).expm1() * denoised_d
            old_denoised = denoised

            if with_images:
                image_list.append(images.clone().detach().cpu())
            if with_x0_images:
                x0_image_list.append(denoised_d.clone().detach().cpu())

        images = images.clamp(-1., 1.)

        if with_images:
            if with_x0_images:
                return unnormalize_to_zero_to_one(images), image_list, x0_image_list
            else:
                return unnormalize_to_zero_to_one(images), image_list
        else:
            return unnormalize_to_zero_to_one(images)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def forward(self, images, condition_x, class_label):
        batch_size, c, h, w, device, image_size, channels = *images.shape, images.device, self.image_size, self.channels

        assert h == image_size and w == image_size, f'height and width of image must be {image_size}'
        assert c == channels, 'mismatch of image channels'

        images = normalize_to_neg_one_to_one(images)
        if torch.randn(1) < self.cond_drop_prob:
            condition_x = None
        else:
            condition_x = normalize_to_neg_one_to_one(condition_x)

        if torch.randn(1) < self.class_cond_drop_prob:
            class_label = None

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1')

        noise = torch.randn_like(images)

        noised_images = images + padded_sigmas * noise  # alphas are 1. in the paper

        denoised = self.preconditioned_network_forward(noised_images, sigmas, condition_x, class_label,
                                                       cond_scale=1.0, class_cond_scale=1.0)

        losses = self.loss_fn(denoised, images, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        losses = losses * self.loss_weight(sigmas)

        return losses.mean()



# neural net helpers

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)

class MonotonicLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return F.linear(x, self.net.weight.abs(), self.net.bias.abs())

# continuous schedules

# equations are taken from https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material
# @crowsonkb Katherine's repository also helped here https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py

# log(snr) that approximates the original linear schedule

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def beta_linear_log_snr(t):
    return -log(expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, s = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5)

class learned_noise_schedule(nn.Module):
    """ described in section H and then I.2 of the supplementary material for variational ddpm paper """

    def __init__(
        self,
        *,
        log_snr_max,
        log_snr_min,
        hidden_dim = 1024,
        frac_gradient = 1.
    ):
        super().__init__()
        self.slope = log_snr_min - log_snr_max
        self.intercept = log_snr_max

        self.net = nn.Sequential(
            Rearrange('... -> ... 1'),
            MonotonicLinear(1, 1),
            Residual(nn.Sequential(
                MonotonicLinear(1, hidden_dim),
                nn.Sigmoid(),
                MonotonicLinear(hidden_dim, 1)
            )),
            Rearrange('... 1 -> ...'),
        )

        self.frac_gradient = frac_gradient

    def forward(self, x):
        frac_gradient = self.frac_gradient
        device = x.device

        out_zero = self.net(torch.zeros_like(x))
        out_one =  self.net(torch.ones_like(x))

        x = self.net(x)

        normed = self.slope * ((x - out_zero) / (out_one - out_zero)) + self.intercept
        return normed * frac_gradient + normed.detach() * (1 - frac_gradient)


class ContinuousTimeGaussianDiffusionSR(nn.Module):
    def set_seed(self, seed):
        torch.cuda.manual_seed(seed)

    def __init__(
        self,
        model,
        *,
        image_size,
        channels = 3,
        noise_schedule = 'linear',
        num_sample_steps = 500,
        clip_sample_denoised = True,
        learned_schedule_net_hidden_dim = 1024,
        learned_noise_schedule_frac_gradient = 1.,   # between 0 and 1, determines what percentage of gradients go back, so one can update the learned noise schedule more slowly
        min_snr_loss_weight = False,
        min_snr_gamma = 5,
        cond_drop_prob = 0.,
        loss_type = 'l2',
    ):
        super().__init__()
        assert model.random_or_learned_sinusoidal_cond
        #assert not model.self_condition, 'not supported yet'

        self.model = model

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # continuous noise schedule related stuff

        if noise_schedule == 'linear':
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == 'cosine':
            self.log_snr = alpha_cosine_log_snr
        elif noise_schedule == 'learned':
            log_snr_max, log_snr_min = [beta_linear_log_snr(torch.tensor([time])).item() for time in (0., 1.)]

            self.log_snr = learned_noise_schedule(
                log_snr_max = log_snr_max,
                log_snr_min = log_snr_min,
                hidden_dim = learned_schedule_net_hidden_dim,
                frac_gradient = learned_noise_schedule_frac_gradient
            )
        else:
            raise ValueError(f'unknown noise schedule {noise_schedule}')

        # sampling

        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised

        # proposed https://arxiv.org/abs/2303.09556

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

        self.cond_drop_prob = cond_drop_prob
        self.loss_type = loss_type

    @property
    def device(self):
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, condition_x, cond_scale, time_next):
        # reviewer found an error in the equation in the paper (missing sigma)
        # following - https://openreview.net/forum?id=2LdBqxc1Yv&noteId=rIQgH0zKsRt

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])
        if cond_scale == 1.0:
            pred_noise = self.model(x, batch_log_snr, condition_x)
        else:
            cond_out = self.model(x, batch_log_snr, condition_x)
            null_out = self.model(x, batch_log_snr, None)
            pred_noise = null_out + (cond_out - null_out) * cond_scale

        x_start = (x - sigma * pred_noise) / alpha

        if self.clip_sample_denoised:
            x_start.clamp_(-1., 1.)
            model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        else:
            model_mean = alpha_next / alpha * (x - c * sigma * pred_noise)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance, x_start

    # sampling related functions

    @torch.inference_mode()
    def p_sample(self, x, time, condition_x, cond_scale, time_next):
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance, x_start = self.p_mean_variance(x = x, time = time, condition_x = condition_x, cond_scale = cond_scale, time_next = time_next)

        if time_next == 0:
            return model_mean, x_start

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise, x_start

    def p_sample_loop(self, shape, condition_x, cond_scale,
                      guidance_start_steps, generation_start_steps, num_sample_steps,
                      with_images, with_x0_images):
        batch = shape[0]

        if generation_start_steps > 0:
            start_time = 1. - torch.tensor(generation_start_steps / num_sample_steps, device=condition_x.device)
            start_times = repeat(start_time, ' -> b', b = batch)
            img, _log_snr = self.q_sample(condition_x, start_times)
        else:
            img = torch.randn(shape, device = self.device)

        if with_images:
            image_list = []
            image_list.append(img.clone().detach().cpu())

        if with_x0_images:
            x0_image_list = []
            x0_image_list.append(img.clone().detach().cpu())

        steps = torch.linspace(1., 0., num_sample_steps + 1, device = self.device)

        for i in tqdm(range(num_sample_steps), desc = 'sampling loop time step', total = num_sample_steps):
            if i < generation_start_steps:
                continue
            if i < guidance_start_steps:
                cur_cond_scale = 1.0
            else:
                cur_cond_scale = cond_scale
            times = steps[i]
            times_next = steps[i + 1]
            with torch.inference_mode():
                img, x_start = self.p_sample(img, times, condition_x, cur_cond_scale, times_next)

            if with_images:
                image_list.append(img.clone().detach().cpu())
            if with_x0_images:
                x0_image_list.append(x_start.clone().detach().cpu())

        img = img.clamp(-1., 1.)
        img = unnormalize_to_zero_to_one(img)

        if with_images:
            if with_x0_images:
                return img, image_list, x0_image_list
            else:
                return img, image_list
        else:
            return img

    @torch.inference_mode()
    def tiled_sample(self, batch_size=4, tile_size=256, tile_stride=256,
                     condition_x=None, class_label=None,
                     cond_scale=1.0, guidance_start_steps=0,
                     class_cond_scale=1.0, class_guidance_start_steps=0,
                     generation_start_steps=0, num_sample_steps=None,
                     with_images=False, with_x0_images=False, start_white_noise=True, amp=False):

        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        condition_x = normalize_to_neg_one_to_one(condition_x)

        batch, c, h, w = condition_x.shape

        # pad condition_x
        coord, pad = get_coord_and_pad(h, w)
        left, top, right, bottom = coord
        condition_x = F.pad(condition_x, pad, mode='reflect')

        if generation_start_steps > 0:
            start_time = 1. - torch.tensor(generation_start_steps / num_sample_steps, device=condition_x.device)
            start_times = repeat(start_time, ' -> b', b = batch)
            img, _log_snr = self.q_sample(condition_x, start_times)
        else:
            if start_white_noise:
                img = torch.randn(condition_x.shape, device = self.device)
            else:
                start_time = torch.tensor(1., device=condition_x.device)
                start_times = repeat(start_time, ' -> b', b = batch)
                img, _log_snr = self.q_sample(condition_x, start_times)

        if with_images:
            image_list = []
            image_list.append(img[:,:,top:bottom,left:right].clone().detach().cpu())

        if with_x0_images:
            x0_image_list = []
            x0_image_list.append(img[:,:,top:bottom,left:right].clone().detach().cpu())

        steps = torch.linspace(1., 0., num_sample_steps + 1, device = self.device)

        # Pre-calculate tile regions
        _, _, height, width = condition_x.shape
        coords0 = get_coords(height, width, tile_size, tile_size, diff=0)
        if height <= tile_size and width <= tile_size:
            coords1 = get_coords(height, width, tile_size, tile_stride, diff=0)
        else:
            coords1 = get_coords(height-tile_size, width-tile_size, tile_size, tile_stride, diff=tile_size//2)
        coord_list = [coords0, coords1]

        # Get the region of the smaller coords
        small_coord, small_pad = get_area(coords1, height, width)
        sleft, stop, sright, sbottom = small_coord

        # Pad the outside of the smaller region of condition_x with 0
        cropped_condition_x = condition_x[:,:,stop:sbottom,sleft:sright]
        condition_x = F.pad(cropped_condition_x, small_pad, mode='constant', value=0)

        x_start = img.clone()

        for i in tqdm(range(num_sample_steps), desc = 'sampling loop time step', total = num_sample_steps):
            if i < generation_start_steps:
                continue
            if i < guidance_start_steps:
                cur_cond_scale = 1.0
            else:
                cur_cond_scale = cond_scale

            times = steps[i]
            times_next = steps[i + 1]

            cur_coords = coord_list[i%2]

            minibatch_index = 0
            minibatch = torch.zeros((batch_size, c, tile_size, tile_size), device=condition_x.device)
            minibatch_condition = torch.zeros((batch_size, c, tile_size, tile_size), device=condition_x.device)
            # minibatch_mask = torch.zeros((batch_size, c, tile_size, tile_size), device=condition_x.device)
            output_indexes = [None] * batch_size
            for hs, he, ws, we in cur_coords:
                minibatch[minibatch_index] = img[:, :, hs:he, ws:we]
                minibatch_condition[minibatch_index] = condition_x[:, :, hs:he, ws:we]
                output_indexes[minibatch_index] = (hs, ws)
                minibatch_index += 1

                if minibatch_index == batch_size:
                    with autocast(enabled=amp):
                        tile_out, tile_x_start = self.p_sample(minibatch, times, minibatch_condition, cur_cond_scale, times_next)
                    for k in range(minibatch_index):
                        hs, ws = output_indexes[k]
                        img[:, :, hs:hs+tile_size, ws:ws+tile_size] = tile_out[k]
                        x_start[:, :, hs:hs+tile_size, ws:ws+tile_size] = tile_x_start[k]
                    minibatch_index = 0

            if minibatch_index > 0:
                with autocast(enabled=amp):
                    tile_out, tile_x_start = self.p_sample(minibatch[0:minibatch_index], times, minibatch_condition[0:minibatch_index], cur_cond_scale, times_next)
                for k in range(minibatch_index):
                    hs, ws = output_indexes[k]
                    img[:, :, hs:hs+tile_size, ws:ws+tile_size] = tile_out[k]
                    x_start[:, :, hs:hs+tile_size, ws:ws+tile_size] = tile_x_start[k]

            if i%2 == 1:
                # Reconstruct by removing the padding part of img when odd times
                cropped_img = img[:,:,stop:sbottom,sleft:sright]
                img, _log_snr = self.q_sample(torch.zeros_like(condition_x), times_next)
                img[:,:,stop:sbottom,sleft:sright] = cropped_img

            if with_images:
                image_list.append(img.clone().detach().cpu())
            if with_x0_images:
                x0_image_list.append(x_start.clone().detach().cpu())


        img = img[:,:,top:bottom,left:right]
        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)

        if with_images:
            if with_x0_images:
                return img, image_list, x0_image_list
            else:
                return img, image_list
        else:
            return img

    def sample(self, batch_size = 16, condition_x = None, cond_scale = 1.0,
               guidance_start_steps = 0, generation_start_steps = 0, num_sample_steps = None,
               with_images=False, with_x0_images=False):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        condition_x = normalize_to_neg_one_to_one(condition_x)
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size), condition_x,
                                  cond_scale, guidance_start_steps, generation_start_steps, num_sample_steps,
                                  with_images, with_x0_images)

    # training related functions - noise prediction

    @autocast(enabled = False)
    def q_sample(self, x_start, times, noise = None, return_alpha_sigma_sum=False):
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised =  x_start * alpha + noise * sigma

        if return_alpha_sigma_sum:
            return x_noised, alpha+sigma
        else:
            return x_noised, log_snr

    def random_times(self, batch_size):
        # times are now uniform from 0 to 1
        return torch.zeros((batch_size,), device = self.device).float().uniform_(0, 1)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, times, condition_x,
                 scaler=None, optimizer=None,
                 discriminator=None, disc_scaler=None, disc_optimizer=None, img_encoder=None,
                 hvd=None, tb_writer=None, global_step=None,
                 noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        batch_size, _, _, _ = x_start.shape

        x, log_snr = self.q_sample(x_start = x_start, times = times, noise = noise)
        model_out = self.model(x, log_snr, condition_x)

        # diffusion loss
        losses = self.loss_fn(model_out, noise, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        losses = losses.mean()

        # TODO: fix
        if self.min_snr_loss_weight:
            snr = log_snr.exp()
            loss_weight = snr.clamp(min = self.min_snr_gamma) / snr
            losses = losses * loss_weight

        return losses

    def forward(self, img, condition_x,
                scaler=None, optimizer=None,
                discriminator=None, disc_scaler=None, disc_optimizer=None, img_encoder=None,
                hvd=None, tb_writer=None, global_step=None,
                *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        times = self.random_times(b)
        img = normalize_to_neg_one_to_one(img)
        condition_x = normalize_to_neg_one_to_one(condition_x)

        if torch.rand(1) < self.cond_drop_prob:
            condition_x = None

        return self.p_losses(img, times, condition_x,
                             scaler, optimizer,
                             discriminator, disc_scaler, disc_optimizer, img_encoder,
                             hvd, tb_writer, global_step,
                             *args, **kwargs)


class ConditionalContinuousTimeGaussianDiffusionSR(nn.Module):
    def set_seed(self, seed):
        torch.cuda.manual_seed(seed)

    def __init__(
        self,
        model,
        *,
        image_size,
        channels = 3,
        noise_schedule = 'linear',
        num_sample_steps = 500,
        clip_sample_denoised = True,
        learned_schedule_net_hidden_dim = 1024,
        learned_noise_schedule_frac_gradient = 1.,   # between 0 and 1, determines what percentage of gradients go back, so one can update the learned noise schedule more slowly
        min_snr_loss_weight = False,
        min_snr_gamma = 5,
        cond_drop_prob = 0.,
        class_cond_drop_prob = 0.,
        loss_type = 'l2',
    ):
        super().__init__()
        assert model.random_or_learned_sinusoidal_cond
        #assert not model.self_condition, 'not supported yet'

        self.model = model

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # continuous noise schedule related stuff

        if noise_schedule == 'linear':
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == 'cosine':
            self.log_snr = alpha_cosine_log_snr
        elif noise_schedule == 'learned':
            log_snr_max, log_snr_min = [beta_linear_log_snr(torch.tensor([time])).item() for time in (0., 1.)]

            self.log_snr = learned_noise_schedule(
                log_snr_max = log_snr_max,
                log_snr_min = log_snr_min,
                hidden_dim = learned_schedule_net_hidden_dim,
                frac_gradient = learned_noise_schedule_frac_gradient
            )
        else:
            raise ValueError(f'unknown noise schedule {noise_schedule}')

        # sampling

        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised

        # proposed https://arxiv.org/abs/2303.09556

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

        self.cond_drop_prob = cond_drop_prob
        self.class_cond_drop_prob = class_cond_drop_prob
        self.loss_type = loss_type

    @property
    def device(self):
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, condition_x, class_label,
                        cond_scale, class_cond_scale, time_next):
        # reviewer found an error in the equation in the paper (missing sigma)
        # following - https://openreview.net/forum?id=2LdBqxc1Yv&noteId=rIQgH0zKsRt

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])

        if (cond_scale != 1.0) and (class_cond_scale != 1.0):
            raise NotImplementedError("Currently, you cannot specify both cond_scale and class_cond_scale at the same time.")
            # full_null_out = self.model(x, batch_log_snr, None, None)
            # class_null_out = self.model(x, batch_log_snr, None, condition_x)
            # cond_null_out = self.model(x, batch_log_snr, class_label, None)
            # full_cond_out = self.model(x, batch_log_snr, class_label, condition_x)
            # pred_noise = full_null_out + \
            #                 ((full_cond_out - class_null_out) * class_cond_scale + \
            #                 (full_cond_out - cond_null_out) * cond_scale) / 2.
        elif cond_scale != 1.0:
            cond_out = self.model(x, batch_log_snr, class_label, condition_x)
            null_out = self.model(x, batch_log_snr, class_label, None)
            pred_noise = null_out + (cond_out - null_out) * cond_scale
        elif class_cond_scale != 1.0:
            cond_out = self.model(x, batch_log_snr, class_label, condition_x)
            null_out = self.model(x, batch_log_snr, None, condition_x)
            pred_noise = null_out + (cond_out - null_out) * class_cond_scale
        elif cond_scale == 1.0 and class_cond_scale == 1.0:
            pred_noise = self.model(x, batch_log_snr, class_label, condition_x)
        else:
            raise NotImplementedError()

        x_start = (x - sigma * pred_noise) / alpha

        if self.clip_sample_denoised:
            x_start.clamp_(-1., 1.)
            model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        else:
            model_mean = alpha_next / alpha * (x - c * sigma * pred_noise)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance, x_start

    # sampling related functions

    @torch.inference_mode()
    def p_sample(self, x, time, condition_x, class_label,
                 cond_scale, class_cond_scale, time_next):
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance, x_start = self.p_mean_variance(x = x, time = time,
                                                                   condition_x = condition_x, class_label = class_label,
                                                                   cond_scale = cond_scale, class_cond_scale = class_cond_scale,
                                                                   time_next = time_next)

        if time_next == 0:
            return model_mean, x_start

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise, x_start

    # @torch.inference_mode()
    def p_sample_loop(self, shape, condition_x, class_label,
                      cond_scale, guidance_start_steps,
                      class_cond_scale, class_guidance_start_steps,
                      generation_start_steps, num_sample_steps,
                      with_images, with_x0_images):
        batch = shape[0]

        if generation_start_steps > 0:
            start_time = 1. - torch.tensor(generation_start_steps / num_sample_steps, device=condition_x.device)
            start_times = repeat(start_time, ' -> b', b = batch)
            img, _log_snr = self.q_sample(condition_x, start_times)
        else:
            img = torch.randn(shape, device = self.device)

        if with_images:
            image_list = []
            image_list.append(img.clone().detach().cpu())

        if with_x0_images:
            x0_image_list = []
            x0_image_list.append(img.clone().detach().cpu())

        steps = torch.linspace(1., 0., num_sample_steps + 1, device = self.device)

        for i in tqdm(range(num_sample_steps), desc = 'sampling loop time step', total = num_sample_steps):
            if i < generation_start_steps:
                continue
            if i < guidance_start_steps:
                cur_cond_scale = 1.0
            else:
                cur_cond_scale = cond_scale
            if i < class_guidance_start_steps:
                cur_class_cond_scale = 1.0
            else:
                cur_class_cond_scale = class_cond_scale
            times = steps[i]
            times_next = steps[i + 1]
            with torch.inference_mode():
                img, x_start = self.p_sample(img, times, condition_x, class_label,
                                            cur_cond_scale, cur_class_cond_scale, times_next)

            if with_images:
                image_list.append(img.clone().detach().cpu())
            if with_x0_images:
                x0_image_list.append(x_start.clone().detach().cpu())

        img = img.clamp(-1., 1.)
        img = unnormalize_to_zero_to_one(img)

        if with_images:
            if with_x0_images:
                return img, image_list, x0_image_list
            else:
                return img, image_list
        else:
            return img

    def delta_p_mean_variance(self, x_start, time, condition_x, class_label,
                              cond_scale, class_cond_scale, time_next, x0):
        # reviewer found an error in the equation in the paper (missing sigma)
        # following - https://openreview.net/forum?id=2LdBqxc1Yv&noteId=rIQgH0zKsRt

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b = x_start.shape[0])

        # Currently not supported for CFG with both condition_x and class_label
        if (cond_scale != 1.0) and (class_cond_scale != 1.0):
            raise NotImplementedError("Currently, you cannot specify both cond_scale and class_cond_scale at the same time.")
        elif cond_scale != 1.0:
            raise NotImplementedError()
        elif class_cond_scale != 1.0:
            raise NotImplementedError()
        elif cond_scale == 1.0 and class_cond_scale == 1.0:
            pred_noise = self.model(x, batch_log_snr, class_label, condition_x)
        else:
            raise NotImplementedError()

        x_start = (x - sigma * pred_noise) / alpha

        if self.clip_sample_denoised:
            x_start.clamp_(-1., 1.)
            model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        else:
            model_mean = alpha_next / alpha * (x - c * sigma * pred_noise)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance, x_start

    def tiled_sample(self, batch_size=4, tile_size=256, tile_stride=256,
                     condition_x=None, class_label=None,
                     cond_scale=1.0, guidance_start_steps=0,
                     class_cond_scale=1.0, class_guidance_start_steps=0,
                     generation_start_steps=0, num_sample_steps=None,
                     with_images=False, with_x0_images=False, start_white_noise=True, amp=False):

        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        condition_x = normalize_to_neg_one_to_one(condition_x)

        batch, c, h, w = condition_x.shape

        # pad condition_x
        coord, pad = get_coord_and_pad(h, w)
        left, top, right, bottom = coord
        condition_x = F.pad(condition_x, pad, mode='reflect')

        if generation_start_steps > 0:
            start_time = 1. - torch.tensor(generation_start_steps / num_sample_steps, device=condition_x.device)
            start_times = repeat(start_time, ' -> b', b = batch)
            img, _log_snr = self.q_sample(condition_x, start_times)
        else:
            if start_white_noise:
                img = torch.randn(condition_x.shape, device = self.device)
            else:
                start_time = torch.tensor(1., device=condition_x.device)
                start_times = repeat(start_time, ' -> b', b = batch)
                img, _log_snr = self.q_sample(condition_x, start_times)

        if with_images:
            image_list = []
            image_list.append(img[:,:,top:bottom,left:right].clone().detach().cpu())

        if with_x0_images:
            x0_image_list = []
            x0_image_list.append(img[:,:,top:bottom,left:right].clone().detach().cpu())

        steps = torch.linspace(1., 0., num_sample_steps + 1, device = self.device)

        # Pre-calculate tile regions
        _, _, height, width = condition_x.shape
        coords0 = get_coords(height, width, tile_size, tile_size, diff=0)
        if height <= tile_size and width <= tile_size:
            coords1 = get_coords(height, width, tile_size, tile_stride, diff=0)
        else:
            coords1 = get_coords(height-tile_size, width-tile_size, tile_size, tile_stride, diff=tile_size//2)
        coord_list = [coords0, coords1]

        # Get the region of the smaller coords
        small_coord, small_pad = get_area(coords1, height, width)
        sleft, stop, sright, sbottom = small_coord

        # Pad the outside of the smaller region of condition_x with 0
        cropped_condition_x = condition_x[:,:,stop:sbottom,sleft:sright]
        condition_x = F.pad(cropped_condition_x, small_pad, mode='constant', value=0)

        x_start = img.clone()

        for i in tqdm(range(num_sample_steps), desc = 'sampling loop time step', total = num_sample_steps):
            if i < generation_start_steps:
                continue
            if i < guidance_start_steps:
                cur_cond_scale = 1.0
            else:
                cur_cond_scale = cond_scale
            if i < class_guidance_start_steps:
                cur_class_cond_scale = 1.0
            else:
                cur_class_cond_scale = class_cond_scale

            times = steps[i]
            times_next = steps[i + 1]

            cur_coords = coord_list[i%2]

            minibatch_index = 0
            minibatch = torch.zeros((batch_size, c, tile_size, tile_size), device=condition_x.device)
            minibatch_condition = torch.zeros((batch_size, c, tile_size, tile_size), device=condition_x.device)
            output_indexes = [None] * batch_size
            for hs, he, ws, we in cur_coords:
                minibatch[minibatch_index] = img[:, :, hs:he, ws:we]
                minibatch_condition[minibatch_index] = condition_x[:, :, hs:he, ws:we]
                output_indexes[minibatch_index] = (hs, ws)
                minibatch_index += 1

                if minibatch_index == batch_size:
                    with torch.inference_mode():
                        tile_out, tile_x_start = self.p_sample(minibatch, times, minibatch_condition, class_label,
                                                               cur_cond_scale, cur_class_cond_scale, times_next)
                    for k in range(minibatch_index):
                        hs, ws = output_indexes[k]
                        img[:, :, hs:hs+tile_size, ws:ws+tile_size] = tile_out[k]
                        x_start[:, :, hs:hs+tile_size, ws:ws+tile_size] = tile_x_start[k]
                    minibatch_index = 0

            if minibatch_index > 0:
                with torch.inference_mode():
                    tile_out, tile_x_start = self.p_sample(minibatch[0:minibatch_index], times, minibatch_condition[0:minibatch_index], class_label,
                                                           cur_cond_scale, cur_class_cond_scale, times_next)
                for k in range(minibatch_index):
                    hs, ws = output_indexes[k]
                    img[:, :, hs:hs+tile_size, ws:ws+tile_size] = tile_out[k]
                    x_start[:, :, hs:hs+tile_size, ws:ws+tile_size] = tile_x_start[k]

            if i%2 == 1:
                # Reconstruct by removing the padding part of img when odd times
                cropped_img = img[:,:,stop:sbottom,sleft:sright]
                img, _log_snr = self.q_sample(torch.zeros_like(condition_x), times_next)
                img[:,:,stop:sbottom,sleft:sright] = cropped_img

            if with_images:
                image_list.append(img.clone().detach().cpu())
            if with_x0_images:
                x0_image_list.append(x_start.clone().detach().cpu())

        img = img[:,:,top:bottom,left:right]
        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)

        if with_images:
            if with_x0_images:
                return img, image_list, x0_image_list
            else:
                return img, image_list
        else:
            return img


    # @torch.inference_mode()
    def sample(self, batch_size = 16, condition_x = None, class_label = None,
               cond_scale = 1.0, guidance_start_steps = 0,
               class_cond_scale = 1.0, class_guidance_start_steps = 0,
               generation_start_steps = 0, num_sample_steps = None,
               with_images=False, with_x0_images=False, x0=None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        condition_x = normalize_to_neg_one_to_one(condition_x)

        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size),
                                  condition_x, class_label,
                                  cond_scale, guidance_start_steps,
                                  class_cond_scale, class_guidance_start_steps,
                                  generation_start_steps, num_sample_steps,
                                  with_images, with_x0_images)

    # training related functions - noise prediction

    @autocast(enabled = False)
    def q_sample(self, x_start, times, noise = None, return_alpha_sigma_sum=False):
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised =  x_start * alpha + noise * sigma

        if return_alpha_sigma_sum:
            return x_noised, alpha+sigma
        else:
            return x_noised, log_snr

    def random_times(self, batch_size):
        # times are now uniform from 0 to 1
        return torch.zeros((batch_size,), device = self.device).float().uniform_(0, 1)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, times, class_label, condition_x, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x, log_snr = self.q_sample(x_start = x_start, times = times, noise = noise)
        model_out = self.model(x, log_snr, class_label, condition_x)

        losses = self.loss_fn(model_out, noise, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        if self.min_snr_loss_weight:
            snr = log_snr.exp()
            loss_weight = snr.clamp(min = self.min_snr_gamma) / snr
            losses = losses * loss_weight

        return losses.mean()

    def forward(self, img, condition_x, class_label, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        times = self.random_times(b)
        img = normalize_to_neg_one_to_one(img)

        if torch.rand(1) < self.cond_drop_prob:
            condition_x = None
        else:
            condition_x = normalize_to_neg_one_to_one(condition_x)

        if torch.rand(1) < self.class_cond_drop_prob:
            class_label = None

        return self.p_losses(img, times, class_label, condition_x, *args, **kwargs)




def get_model(conf, logger):
    dim_mults = tuple([int(elem) for elem in conf.ddpm_unet_dim_mults.split(',')])
    full_attn = tuple([eval(elem) for elem in conf.full_attn.split(',')])
    if 'conditional' in conf.model:
        unet = ConditionalSRUnet(
            dim = conf.unet_dim,
            dim_mults = dim_mults,
            full_attn = full_attn,
            learned_variance = conf.learned_variance,
            learned_sinusoidal_cond = conf.learned_sinusoidal_cond,
            learned_sinusoidal_dim = conf.learned_sinusoidal_dim,
            flash_attn = conf.flash_attn,
            pixel_shuffle_upsample = conf.pixel_shuffle_upsample,
            num_classes = conf.num_classes
        )
        logger.info(f"ConditionalSRUnet: channels=6 dim={conf.unet_dim} dim_mults={conf.ddpm_unet_dim_mults} num_classes={conf.num_classes}")
    else:
        unet = SRUnet(
            dim = conf.unet_dim,
            dim_mults = dim_mults,
            full_attn = full_attn,
            learned_variance = conf.learned_variance,
            learned_sinusoidal_cond = conf.learned_sinusoidal_cond,
            learned_sinusoidal_dim = conf.learned_sinusoidal_dim,
            flash_attn = conf.flash_attn,
            pixel_shuffle_upsample = conf.pixel_shuffle_upsample,
            use_free_u = conf.use_free_u,
            free_u_b1 = conf.free_u_b1,
            free_u_b2 = conf.free_u_b2,
            free_u_s1 = conf.free_u_s1,
            free_u_s2 = conf.free_u_s2
        )
        logger.info(f"SRUnet: channels=6 dim={conf.unet_dim} dim_mults={conf.ddpm_unet_dim_mults}")

    if conf.model == 'gaussian':
        assert not conf.learned_sinusoidal_cond
        conf.use_dpmpp_solver = False
        model = GaussianDiffusionSR(
            model = unet,
            image_size = conf.image_size,
            timesteps = conf.timesteps,
            sampling_timesteps = conf.sampling_timesteps,
            objective = conf.objective,
            beta_schedule = conf.beta_schedule,
            offset_noise_strength = conf.offset_noise_strength,
            min_snr_loss_weight = conf.min_snr_loss_weight,
            min_snr_gamma = conf.min_snr_gamma,
            cond_drop_prob = conf.cond_drop_prob,
            loss_type = conf.loss_type,
        )
        logger.info(f"GaussianDiffusionSR: image_size={conf.image_size} timesteps={conf.timesteps} sampling_timesteps={conf.sampling_timesteps}")

    elif conf.model == 'conditional_gaussian':
        assert not conf.learned_sinusoidal_cond
        conf.use_dpmpp_solver = False
        model = ConditionalGaussianDiffusionSR(
            model = unet,
            image_size = conf.image_size,
            timesteps = conf.timesteps,
            sampling_timesteps = conf.sampling_timesteps,
            objective = conf.objective,
            beta_schedule = conf.beta_schedule,
            offset_noise_strength = conf.offset_noise_strength,
            min_snr_loss_weight = conf.min_snr_loss_weight,
            min_snr_gamma = conf.min_snr_gamma,
            cond_drop_prob = conf.cond_drop_prob,
            class_cond_drop_prob = conf.class_cond_drop_prob,
            loss_type = conf.loss_type,
        )
        logger.info(f"ConditionalGaussianDiffusionSR: image_size={conf.image_size} timesteps={conf.timesteps} sampling_timesteps={conf.sampling_timesteps}")

    elif conf.model == 'elucidated':
        assert conf.learned_sinusoidal_cond
        model = ElucidatedDiffusionSR(
            net = unet,
            image_size = conf.image_size,
            num_sample_steps = conf.num_sample_steps,
            sigma_min = conf.sigma_min,
            sigma_max = conf.sigma_max,
            sigma_data = conf.sigma_data,
            rho = conf.rho,
            P_mean = conf.P_mean,
            P_std = conf.P_std,
            S_churn = conf.S_churn,
            S_tmin = conf.S_tmin,
            S_tmax = conf.S_tmax,
            S_noise = conf.S_noise,
            cond_drop_prob = conf.cond_drop_prob,
            use_dpmpp_solver = conf.use_dpmpp_solver,
            loss_type = conf.loss_type
        )
        logger.info(f"ElucidatedDiffusionSR: image_size={conf.image_size} num_sample_steps={conf.num_sample_steps}")

    elif conf.model == 'conditional_elucidated':
        assert conf.learned_sinusoidal_cond
        model = ConditionalElucidatedDiffusionSR(
            net = unet,
            image_size = conf.image_size,
            num_sample_steps = conf.num_sample_steps,
            sigma_min = conf.sigma_min,
            sigma_max = conf.sigma_max,
            sigma_data = conf.sigma_data,
            rho = conf.rho,
            P_mean = conf.P_mean,
            P_std = conf.P_std,
            S_churn = conf.S_churn,
            S_tmin = conf.S_tmin,
            S_tmax = conf.S_tmax,
            S_noise = conf.S_noise,
            cond_drop_prob = conf.cond_drop_prob,
            class_cond_drop_prob = conf.class_cond_drop_prob,
            use_dpmpp_solver = conf.use_dpmpp_solver,
            loss_type = conf.loss_type
        )
        logger.info(f"ConditionalElucidatedDiffusionSR: image_size={conf.image_size} num_sample_steps={conf.num_sample_steps}")

    elif conf.model == 'continuous':
        assert conf.learned_sinusoidal_cond
        conf.use_dpmpp_solver = False
        model = ContinuousTimeGaussianDiffusionSR(
            model = unet,
            image_size = conf.image_size,
            noise_schedule = conf.noise_schedule,
            num_sample_steps = conf.num_sample_steps,
            clip_sample_denoised = conf.clip_sample_denoised,
            learned_schedule_net_hidden_dim = conf.learned_schedule_net_hidden_dim,
            learned_noise_schedule_frac_gradient = conf.learned_noise_schedule_frac_gradient,
            min_snr_loss_weight = conf.min_snr_loss_weight,
            min_snr_gamma = conf.min_snr_gamma,
            cond_drop_prob = conf.cond_drop_prob,
            loss_type = conf.loss_type,
        )
        logger.info(f"ContinuousTimeGaussianDiffusionSR: image_size={conf.image_size} num_sample_steps={conf.num_sample_steps}")

    elif conf.model == 'conditional_continuous':
        assert conf.learned_sinusoidal_cond
        conf.use_dpmpp_solver = False
        model = ConditionalContinuousTimeGaussianDiffusionSR(
            model = unet,
            image_size = conf.image_size,
            noise_schedule = conf.noise_schedule,
            num_sample_steps = conf.num_sample_steps,
            clip_sample_denoised = conf.clip_sample_denoised,
            learned_schedule_net_hidden_dim = conf.learned_schedule_net_hidden_dim,
            learned_noise_schedule_frac_gradient = conf.learned_noise_schedule_frac_gradient,
            min_snr_loss_weight = conf.min_snr_loss_weight,
            min_snr_gamma = conf.min_snr_gamma,
            cond_drop_prob = conf.cond_drop_prob,
            class_cond_drop_prob = conf.class_cond_drop_prob,
            loss_type = conf.loss_type,
        )
        logger.info(f"ConditionalContinuousTimeGaussianDiffusionSR: image_size={conf.image_size} num_sample_steps={conf.num_sample_steps}")

    else:
        raise NotImplementedError(conf.model)

    # ema model
    ema_model = ModelEmaV2(model, decay=conf.ema_decay)

    if conf.ckpt_path:
        ckpt = torch.load(conf.ckpt_path, map_location='cpu', weights_only=True)
        # ema model
        check = ema_model.module.load_state_dict(ckpt['ema_model'], strict=conf.load_strict)
        logger.info(f"load ema_model weight from : {conf.ckpt_path}")
        logger.info(f"check: {check}")

    return ema_model


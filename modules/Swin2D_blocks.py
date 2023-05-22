##### 3D video Swin transformer not official #####
##### Xupeng Chen, 11/11/2022 #####


import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_


###########################3D SWIN TRANSFORMER##########################
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

'''
class Mlp(nn.Module):
    #No Need to Change to 3D
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
    
def window_partition(x, window_size_T, window_size_E):
    #   Need to Change to 3D
    """
    Args:
        x: (B, T, H, W, C)
        window_size (int): window_size_T, window_size_E
    Returns:
        windows: (num_windows*B, window_size_T, window_size_E, window_size_E, C), num_windows = (T// window_size_T) * (H // window_size_E) * (W // window_size_E)
    """
    B, T, H, W, C = x.shape
    #x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.view(B, T// window_size_T, window_size_T, H // window_size_E, window_size_E, W // window_size_E, window_size_E, C)
    #          0, 1                , 2            , 3                 , 4            , 5                 ,6             , 7
    #windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5,  2, 4, 6, 7).contiguous().view(-1, window_size_T, window_size_E, window_size_E, C)
    return windows


def window_reverse(windows, window_size_T, window_size_E, T, H, W):
    #   Need to Change to 3D
    """
    Args:
        windows: (num_windows*B, window_size_T, window_size_E, window_size_E, C)
        window_size (int): window_size_T, window_size_E
        T (int): Temporal length of the video/3D volume
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, T, H, W, C)
    """
    B = int(windows.shape[0] / (T* H * W / window_size_T / window_size_E / window_size_E))
    #x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = windows.view(B, T // window_size_T ,  H // window_size_E, W // window_size_E, window_size_T, window_size_E, window_size_E, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, -1)
    #x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    #   Need to Change to 3D
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size_T, window_size_E, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size_T = window_size_T  # Wt
        self.window_size_E = window_size_E  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size_T - 1) * (2 * window_size_E[0] - 1) * (2 * window_size_E[1] - 1), num_heads))  # 2*Wt-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_t = torch.arange(self.window_size_T)
        coords_h = torch.arange(self.window_size_E[0])
        coords_w = torch.arange(self.window_size_E[1])
        coords = torch.stack(torch.meshgrid(coords_t,coords_h,coords_w))  # 3, Wt, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wt*Wh*Ww
        #print ('coords_flatten', coords_flatten.shape)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wt*Wh*Ww, Wt*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wt*Wh*Ww, Wt*Wh*Ww, 3
        relative_coords[:, :, 0] +=  self.window_size_T - 1  # shift to start from 0
        relative_coords[:, :, 1] +=  self.window_size_E[0] - 1 
        relative_coords[:, :, 2] +=  self.window_size_E[1] - 1
        relative_coords[:, :, 0] *= 4 *  self.window_size_T - 1
        relative_coords[:, :, 1] *= 2 *  self.window_size_E[0] - 1
        relative_position_index = relative_coords.sum(-1)  # Wt*Wh*Ww, Wt*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            N: Wt*Wh*Ww
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        #print ('q,k,v', q.shape, k.shape, v.shape)
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size_T * self.window_size_E[0] * self.window_size_E[1], self.window_size_T * self.window_size_E[0] * self.window_size_E[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wt*Wh*Ww, Wt*Wh*Ww
        #print ('attn', attn.shape,relative_position_bias.unsqueeze(0).shape)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size_T={self.window_size_T}, window_size_E={self.window_size_E},num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class PatchMerging(nn.Module):
    #   Need to Change to 3D
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        temporal_down: Default true, but we should allow false to downsample electrode only
        elec_down: Default true, but we should allow false to downsample temporal only since our input temporal length>> electrodes length
    """

    def __init__(self,  input_resolution, dim, norm_layer=nn.LayerNorm, temporal_down=True, elec_down=True):
        super().__init__()
        #self.input_temporal = input_temporal
        self.input_resolution = input_resolution
        self.dim = dim
        self.temporal_down = temporal_down
        self.elec_down = elec_down
        temp_downfactor = 2 if self.temporal_down else 1
        elec_downfactor = 4 if self.elec_down else 1
        self.reduction = nn.Linear( dim * elec_downfactor * temp_downfactor, 2 * dim, bias=False)
        self.norm = norm_layer( dim * elec_downfactor * temp_downfactor)
        

    def forward(self, x):
        """
        x: B, T*H*W, C
        """
        #T = self.input_temporal
        T, H, W = self.input_resolution
        B, L, C = x.shape
        #print (L, T * H * W, T, H, W)
        assert L == T * H * W, "input feature has wrong size"
        assert T%2 ==0 and H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, T, H, W, C)
        #print (x.shape)
        # sample different part of the image, concat, then reduce C dimension, thus we still have all the information but H and W reduced!
        # if 3D, we have 8 combinations:
        
        #x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        #x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        #x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        #x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        #x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        #x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        if self.temporal_down and self.elec_down:
            #print ('self.temporal_down and self.elec_down',self.temporal_down, self.elec_down)
            x0 = x[:, 0::2, 0::2, 0::2, :]  # B T/2 H/2 W/2 C
            x1 = x[:, 0::2, 1::2, 0::2, :]  # B T/2 H/2 W/2 C
            x2 = x[:, 0::2, 0::2, 1::2, :]  # B T/2 H/2 W/2 C
            x3 = x[:, 0::2, 1::2, 1::2, :]  # B T/2 H/2 W/2 C
            x4 = x[:, 1::2, 0::2, 0::2, :]  # B T/2 H/2 W/2 C
            x5 = x[:, 1::2, 1::2, 0::2, :]  # B T/2 H/2 W/2 C
            x6 = x[:, 1::2, 0::2, 1::2, :]  # B T/2 H/2 W/2 C
            x7 = x[:, 1::2, 1::2, 1::2, :]  # B T/2 H/2 W/2 C
            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B T/2 H/2 W/2 8*C
            x = x.view(B, -1, 8 * C)  # B T/2*H/2*W/2 8*C
        elif self.temporal_down and (not self.elec_down):
            #print ('self.temporal_down and self.elec_down',self.temporal_down, self.elec_down)
            x0 = x[:, 0::2, :, :, :]  # B T/2 H W C
            x1 = x[:, 1::2, :, :, :]  # B T/2 H W C
            x = torch.cat([x0, x1], -1)  # B T/2 H W 2*C
            x = x.view(B, -1, 2 * C)  # B T/2*H *W 2*C
        elif (not self.temporal_down) and self.elec_down:
            #print ('self.temporal_down and self.elec_down',self.temporal_down, self.elec_down)
            x0 = x[:, :, 0::2, 0::2, :]  # B T H/2 W/2 C
            x1 = x[:, :, 1::2, 0::2, :]  # B T H/2 W/2 C
            x2 = x[:, :, 0::2, 1::2, :]  # B T H/2 W/2 C
            x3 = x[:, :, 1::2, 1::2, :]  # B T H/2 W/2 C
            x = torch.cat([x0, x1, x2, x3], -1)  # B T  H/2 W/2 4*C
            x = x.view(B, -1, 4 * C)  # B T *H/2*W/2 4*C
        else: #no patch merging at all
            #print ('self.temporal_down and self.elec_down',self.temporal_down, self.elec_down)
            x = x
            x = x.view(B, -1,  C)  # B T *H *W C
        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        T, H, W = self.input_resolution
        flops = T* H * W * self.dim
        flops += (T // 2) * (H // 2) * (W // 2) * 8 * self.dim * 2 * self.dim
        return flops

class PatchEmbed(nn.Module):
    #   Need to Change to 3D
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 16
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 1
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    Returns:
        Downsampled Image(3D volume): B,T//patch_size_T* H//patch_size_E* W//patch_size_E,C   
    """

    def __init__(self, temporal_size=128, img_size=16, patch_size_T=4, patch_size_E=4, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (temporal_size, img_size, img_size)#to_2tuple(img_size)
        patch_size = (patch_size_T, patch_size_E, patch_size_E)#to_2tuple(patch_size)
        #(32, 4, 4)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
        self.temporal_size= temporal_size 
        self.img_size = img_size
        self.patch_size = patch_size_E
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, T, H, W = x.shape #previously  B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert T == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            f"Input image size ({T}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        #print ('self.proj(x)',self.proj(x).shape)
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Pt*Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        To, Ho, Wo = self.patches_resolution
        flops = To * Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        if self.norm is not None:
            flops += To * Ho * Wo * self.embed_dim
        return flops

class SwinTransformerBlock(nn.Module):
    #   Need to Change to 3D
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size_T=8, window_size_E=4, shift_size_T=0,shift_size_E=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size_T = window_size_T
        self.window_size_E = window_size_E
        self.shift_size_T = shift_size_T
        self.shift_size_E = shift_size_E
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size_E:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size_E < self.window_size_E, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size_T, (window_size_E, window_size_E), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size_E > 0:
            # calculate attention mask for SW-MSA
            T, H, W =  input_resolution
            img_mask = torch.zeros((1, T, H, W, 1))  # 1 H W 1
            t_slices = (slice(0, - window_size_T),
                        slice(- window_size_T, - shift_size_T),
                        slice(- shift_size_T, None))
            h_slices = (slice(0, - window_size_E),
                        slice(- window_size_E, - shift_size_E),
                        slice(- shift_size_E, None))
            w_slices = (slice(0, - window_size_E),
                        slice(- window_size_E, - shift_size_E),
                        slice(- shift_size_E, None))
            cnt = 0
            for t in t_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, t, h, w, :] = cnt
                        cnt += 1

            mask_windows = window_partition(img_mask,  window_size_T, window_size_E)  # nW, window_size, window_size, 1
            #print ('mask_windows.shape', mask_windows.shape)
            mask_windows = mask_windows.view(-1,  window_size_T * window_size_E *  window_size_E)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        #print ('in block',x.shape)
        T, H, W = self.input_resolution
        B, L, C = x.shape
        #print ('L, T* H * W, T, H, W',L, T* H * W, T, H, W)
        assert L == T* H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, T, H, W, C)
        #print ('x.view(B, T, H, W, C)',x.shape)
        # cyclic shift
        if self.shift_size_E > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size_T, -self.shift_size_E, -self.shift_size_E),\
                                   dims=(1, 2, 3)) #T, H, W dimension shift
        else:
            shifted_x = x
        #print ('shifted_x',shifted_x.shape)
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size_T, self.window_size_E)  # nW*B, window_size_T, window_size_E, window_size_E, C
        #print ('x_windows',x_windows.shape)
        x_windows = x_windows.view(-1, self.window_size_T * self.window_size_E * self.window_size_E, C)  
        # nW*B, window_size_T*window_size_E*window_size_E, C
        #print ('x_windows',x_windows.shape)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size_T, self.window_size_E, self.window_size_E, C)
        shifted_x = window_reverse(attn_windows, self.window_size_T, self.window_size_E, T, H, W)  # B T' H' W' C
        #print ('attn_windows, shifted_x', attn_windows.shape, shifted_x.shape)
        
        # reverse cyclic shift
        if self.shift_size_E > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size_T, self.shift_size_E, self.shift_size_E), dims=(1, 2, 3))
        else:
            x = shifted_x
        #print ('x shape', x.shape)
        x = x.view(B, T* H * W, C)
        #print ('x view',x.shape)
    
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        #print ('out block', x.shape)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size_T={self.window_size_T}, window_size_E={self.window_size_E}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops
class BasicLayer(nn.Module):
    #Need to Change to 3D
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size_T,  window_size_E,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                temporal_down=True,elec_down=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size_T=window_size_T, window_size_E=window_size_E,
                                 shift_size_E=0 if (i % 2 == 0) else window_size_E // 2,
                                 shift_size_T=0 if (i % 2 == 0) else window_size_T // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer,
                                        temporal_down=temporal_down, elec_down=elec_down)
        else:
            self.downsample = None

    def forward(self, x):
        #print ('basic layer before blocks',x.shape)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        #print ('basic layer after blocks',x.shape)
        if self.downsample is not None:
            x = self.downsample(x)
        #print ('basic layer after downsample',x.shape)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class SwinTransformer(nn.Module):
    #No Need to Change to 3D
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, temporal_size=128, img_size=16, patch_size_T=2, patch_size_E=2, in_chans=1, num_classes=64,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size_T=4, window_size_E=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.temporal_size = temporal_size
        self.img_size = img_size
        # split image into non-overlapping patches
        #temporal_size=128, img_size=16, patch_size_T=4, patch_size_E=4
        self.patch_embed = PatchEmbed(temporal_size=temporal_size,
            img_size=img_size, patch_size_T=patch_size_T, patch_size_E=patch_size_E, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        #print ('patches_resolution', patches_resolution)
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        
        #only downsample twice/three times
        downsample_time = self.num_layers - 1
        
        downsample_elec = [True for i_layer in range(self.num_layers)]
        if patch_size_E >=2:
            #if self.num_layers >=3:
                for i_layer in range(1, self.num_layers):
                    downsample_elec[i_layer] = False #which means we only downsample electrodes three times, then we only do temporal downsample if we have more depths
        else:
            #if self.num_layers >=4:
                for i_layer in range(2, self.num_layers):
                    downsample_elec[i_layer] = False #which means we only downsample electrodes three times, then we only do temporal downsample if we have more depths
        #E_downsample_times = np.prod([2**i for i in downsample_elec])
        
        for i_layer in range(self.num_layers):
            #print ('depths[:i_layer',depths, i_layer)
            #print ('sum',sum(depths[:i_layer]),sum(depths[:i_layer + 1]))
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // int(np.prod([2**downfact for downfact in downsample_elec[:i_layer]])),#(2 ** i_layer),
                                                 patches_resolution[2] // int(np.prod([2**downfact for downfact in downsample_elec[:i_layer]])) ),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size_T=window_size_T,
                               window_size_E=window_size_E,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[int(sum(depths[:i_layer])):int(sum(depths[:i_layer + 1]))],
                               norm_layer=norm_layer,
                               #downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               downsample=PatchMerging if (i_layer < downsample_time) else None, #downsample only twice, if no patchembeding (patchsize=1), then 3 times
                               use_checkpoint=use_checkpoint,
                               temporal_down=True,#temporal_down, 
                               elec_down=downsample_elec[i_layer])
            self.layers.append(layer)
        #print ('E_down',int(np.prod([2**downfact for downfact in downsample_elec])))
        self.E_down = int(np.prod([2**downfact for downfact in downsample_elec]))*patch_size_E #final downsample times for elec
        self.T_down = 2**downsample_time*patch_size_T #final downsample times for temporal
        #print ('E_down, T_down',self.E_down, self.T_down,np.prod([2**downfact for downfact in downsample_elec]),patch_size_E)
        
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        #self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        #print ('head feature dim',img_size//(2**len(depths))*embed_dim*((len(depths)-1)**2),img_size//(len(depths)**2), embed_dim,(2**(len(depths)-1)))
        #in_embed = img_size//(2**len(depths) )*img_size//(2**len(depths) ) *embed_dim*(2**(len(depths)-1))*(2**(len(depths)-1))
        in_embed = img_size//( self.E_down )*img_size//( self.E_down ) *embed_dim*(2**(downsample_time-1))*(2**(downsample_time-1)) //2**(self.num_layers-3)
        self.head = nn.Linear(in_embed, num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        #print ('before patch embed', x.shape)
        x = self.patch_embed(x)
        #print ('after patch embed', x.shape)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        #print (x.shape)
        for layer in  self.layers:
            #print ('layer',layer)
            #print ('x shape before layer',x.shape)
            x = layer(x)
            #print ('x shape after layer',x.shape)
        #print (x.shape)
        x = self.norm(x)  # B L C
        #x = self.avgpool(x.transpose(1, 2))  # B C 1
        #print (x.shape)
        #x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        
        B, THW, C = x.shape
        T = self.temporal_size//self.T_down
        H = self.img_size//self.E_down
        #print ('T, H', T, H)
        #print ('B, THW, C', B, THW, C)
        # original 128(16*8), 16, 16
        #H = int(HW**0.5)
        x = x.view(B, T, H, H, C)
        #print ('x, B, T, H, H, C',x.shape)
        x = x.view(B, T, H* H *C)
        #print ('xview', x.shape)
        #print ('head', self.head)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
'''

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    # No Need to Change to 3D
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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


def window_partition(x, window_size_T, window_size_E):
    #   Need to Change to 3D
    """
    Args:
        x: (B, T, H, W, C)
        window_size (int): window_size_T, window_size_E
    Returns:
        windows: (num_windows*B, window_size_T, window_size_E, window_size_E, C), num_windows = (T// window_size_T) * (H // window_size_E) * (W // window_size_E)
    """
    B, T, H, W, C = x.shape
    # x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.view(
        B,
        T // window_size_T,
        window_size_T,
        H // window_size_E[0],
        window_size_E[0],
        W // window_size_E[1],
        window_size_E[1],
        C,
    )
    #          0, 1                , 2            , 3                 , 4            , 5                 ,6             , 7
    # windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .contiguous()
        .view(-1, window_size_T, window_size_E[0], window_size_E[1], C)
    )
    return windows


def window_reverse(windows, window_size_T, window_size_E, T, H, W):
    #   Need to Change to 3D
    """
    Args:
        windows: (num_windows*B, window_size_T, window_size_E, window_size_E, C)
        window_size (int): window_size_T, window_size_E
        T (int): Temporal length of the video/3D volume
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, T, H, W, C)
    """
    B = int(
        windows.shape[0]
        / (T * H * W / window_size_T / window_size_E[0] / window_size_E[1])
    )
    # x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = windows.view(
        B,
        T // window_size_T,
        H // window_size_E[0],
        W // window_size_E[1],
        window_size_T,
        window_size_E[0],
        window_size_E[1],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, -1)
    # x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# no need to change for non square
class WindowAttention(nn.Module):
    #   Need to Change to 3D
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size_T,
        window_size_E,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size_T = window_size_T  # Wt
        self.window_size_E = window_size_E  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size_T - 1)
                * (2 * window_size_E[0] - 1)
                * (2 * window_size_E[1] - 1),
                num_heads,
            )
        )  # 2*Wt-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_t = torch.arange(self.window_size_T)
        coords_h = torch.arange(self.window_size_E[0])
        coords_w = torch.arange(self.window_size_E[1])
        coords = torch.stack(
            torch.meshgrid(coords_t, coords_h, coords_w)
        )  # 3, Wt, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wt*Wh*Ww
        # print ('coords_flatten', coords_flatten.shape)
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 3, Wt*Wh*Ww, Wt*Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wt*Wh*Ww, Wt*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size_T - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size_E[0] - 1
        relative_coords[:, :, 2] += self.window_size_E[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size_T - 1
        relative_coords[:, :, 1] *= 2 * self.window_size_E[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wt*Wh*Ww, Wt*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            N: Wt*Wh*Ww
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        # print ('q,k,v', q.shape, k.shape, v.shape)
        attn = q @ k.transpose(-2, -1)
        # print (self.relative_position_bias_table.shape)
        # print ('view',self.relative_position_index.shape, self.relative_position_index.view(-1).shape,self.window_size_T * self.window_size_E[0] * self.window_size_E[1], self.window_size_T * self.window_size_E[0] * self.window_size_E[1])
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size_T * self.window_size_E[0] * self.window_size_E[1],
            self.window_size_T * self.window_size_E[0] * self.window_size_E[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wt*Wh*Ww, Wt*Wh*Ww
        # print ('attn', attn.shape,relative_position_bias.unsqueeze(0).shape)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size_T={self.window_size_T}, window_size_E={self.window_size_E},num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class PatchMerging(nn.Module):
    #   Need to Change to 3D
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        temporal_down: Default true, but we should allow false to downsample electrode only
        elec_down: Default true, but we should allow false to downsample temporal only since our input temporal length>> electrodes length
    """

    def __init__(
        self,
        input_resolution,
        dim,
        norm_layer=nn.LayerNorm,
        temporal_down=True,
        elec_down=True,
    ):
        super().__init__()
        # self.input_temporal = input_temporal
        self.input_resolution = input_resolution
        self.dim = dim
        self.temporal_down = temporal_down
        self.elec_down = elec_down
        temp_downfactor = 2 if self.temporal_down else 1
        elec_downfactor = 4 if self.elec_down else 1
        self.reduction = nn.Linear(
            dim * elec_downfactor * temp_downfactor, 2 * dim, bias=False
        )
        self.norm = norm_layer(dim * elec_downfactor * temp_downfactor)

    def forward(self, x):
        """
        x: B, T*H*W, C
        """
        # T = self.input_temporal
        T, H, W = self.input_resolution
        B, L, C = x.shape
        # print (L, T * H * W, T, H, W)
        assert L == T * H * W, "input feature has wrong size"
        assert (
            T % 2 == 0 and H % 2 == 0 and W % 2 == 0
        ), f"x size ({H}*{W}) are not even."

        x = x.view(B, T, H, W, C)
        # print (x.shape)
        # sample different part of the image, concat, then reduce C dimension, thus we still have all the information but H and W reduced!
        # if 3D, we have 8 combinations:

        # x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        if self.temporal_down and self.elec_down:
            # print ('self.temporal_down and self.elec_down',self.temporal_down, self.elec_down)
            x0 = x[:, 0::2, 0::2, 0::2, :]  # B T/2 H/2 W/2 C
            x1 = x[:, 0::2, 1::2, 0::2, :]  # B T/2 H/2 W/2 C
            x2 = x[:, 0::2, 0::2, 1::2, :]  # B T/2 H/2 W/2 C
            x3 = x[:, 0::2, 1::2, 1::2, :]  # B T/2 H/2 W/2 C
            x4 = x[:, 1::2, 0::2, 0::2, :]  # B T/2 H/2 W/2 C
            x5 = x[:, 1::2, 1::2, 0::2, :]  # B T/2 H/2 W/2 C
            x6 = x[:, 1::2, 0::2, 1::2, :]  # B T/2 H/2 W/2 C
            x7 = x[:, 1::2, 1::2, 1::2, :]  # B T/2 H/2 W/2 C
            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B T/2 H/2 W/2 8*C
            x = x.view(B, -1, 8 * C)  # B T/2*H/2*W/2 8*C
        elif self.temporal_down and (not self.elec_down):
            # print ('self.temporal_down and self.elec_down',self.temporal_down, self.elec_down)
            x0 = x[:, 0::2, :, :, :]  # B T/2 H W C
            x1 = x[:, 1::2, :, :, :]  # B T/2 H W C
            x = torch.cat([x0, x1], -1)  # B T/2 H W 2*C
            x = x.view(B, -1, 2 * C)  # B T/2*H *W 2*C
        elif (not self.temporal_down) and self.elec_down:
            # print ('self.temporal_down and self.elec_down',self.temporal_down, self.elec_down)
            x0 = x[:, :, 0::2, 0::2, :]  # B T H/2 W/2 C
            x1 = x[:, :, 1::2, 0::2, :]  # B T H/2 W/2 C
            x2 = x[:, :, 0::2, 1::2, :]  # B T H/2 W/2 C
            x3 = x[:, :, 1::2, 1::2, :]  # B T H/2 W/2 C
            x = torch.cat([x0, x1, x2, x3], -1)  # B T  H/2 W/2 4*C
            x = x.view(B, -1, 4 * C)  # B T *H/2*W/2 4*C
        else:  # no patch merging at all
            # print ('self.temporal_down and self.elec_down',self.temporal_down, self.elec_down)
            x = x
            x = x.view(B, -1, C)  # B T *H *W C
        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        T, H, W = self.input_resolution
        flops = T * H * W * self.dim
        flops += (T // 2) * (H // 2) * (W // 2) * 8 * self.dim * 2 * self.dim
        return flops


class PatchEmbed(nn.Module):
    #   Need to Change to 3D
    r"""Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 16
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 1
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    Returns:
        Downsampled Image(3D volume): B,T//patch_size_T* H//patch_size_E* W//patch_size_E,C
    """

    def __init__(
        self,
        temporal_size=128,
        img_size=16,
        patch_size_T=4,
        patch_size_E=(4, 4),
        in_chans=1,
        embed_dim=96,
        norm_layer=None,
    ):
        super().__init__()
        img_size = (temporal_size, img_size, img_size)  # to_2tuple(img_size)
        patch_size = (
            patch_size_T,
            patch_size_E[0],
            patch_size_E[1],
        )  # to_2tuple(patch_size)
        # (32, 4, 4)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        ]
        self.temporal_size = temporal_size
        self.img_size = img_size
        # self.patch_size = patch_size_E
        self.patches_resolution = patches_resolution
        self.num_patches = (
            patches_resolution[0] * patches_resolution[1] * patches_resolution[2]
        )

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, T, H, W = x.shape  # previously  B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            T == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2]
        ), f"Input image size ({T}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        # print ('self.proj(x)',self.proj(x).shape)
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Pt*Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        To, Ho, Wo = self.patches_resolution
        flops = (
            To
            * Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        )
        if self.norm is not None:
            flops += To * Ho * Wo * self.embed_dim
        return flops


# non-square version
class SwinTransformerBlock(nn.Module):
    #   Need to Change to 3D
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size_T=8,
        window_size_E=4,
        shift_size_T=0,
        shift_size_E=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size_T = window_size_T
        self.window_size_E = window_size_E
        self.shift_size_T = shift_size_T
        self.shift_size_E = shift_size_E
        self.mlp_ratio = mlp_ratio
        if self.input_resolution[0] <= self.window_size_E[0]:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size_E = (0, 0)
            self.window_size_E = (self.input_resolution[0], self.input_resolution[1])
        assert (
            0 <= self.shift_size_E[0] < self.window_size_E[0]
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size_T,
            (window_size_E[0], window_size_E[1]),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size_E[0] > 0:
            # calculate attention mask for SW-MSA
            T, H, W = input_resolution
            img_mask = torch.zeros((1, T, H, W, 1))  # 1 H W 1
            t_slices = (
                slice(0, -window_size_T),
                slice(-window_size_T, -shift_size_T),
                slice(-shift_size_T, None),
            )
            h_slices = (
                slice(0, -window_size_E[0]),
                slice(-window_size_E[0], -shift_size_E[0]),
                slice(-shift_size_E[0], None),
            )
            w_slices = (
                slice(0, -window_size_E[1]),
                slice(-window_size_E[1], -shift_size_E[1]),
                slice(-shift_size_E[1], None),
            )
            cnt = 0
            for t in t_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, t, h, w, :] = cnt
                        cnt += 1

            mask_windows = window_partition(
                img_mask, window_size_T, window_size_E
            )  # nW, window_size, window_size, 1
            # print ('mask_windows.shape', mask_windows.shape)
            mask_windows = mask_windows.view(
                -1, window_size_T * window_size_E[0] * window_size_E[1]
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # print ('in block',x.shape)
        T, H, W = self.input_resolution
        B, L, C = x.shape
        # print ('L, T* H * W, T, H, W',L, T* H * W, T, H, W)
        assert L == T * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, T, H, W, C)
        # print ('x.view(B, T, H, W, C)',x.shape)
        # cyclic shift
        if self.shift_size_E[0] > 0:
            shifted_x = torch.roll(
                x,
                shifts=(
                    -self.shift_size_T,
                    -self.shift_size_E[0],
                    -self.shift_size_E[1],
                ),
                dims=(1, 2, 3),
            )  # T, H, W dimension shift
        else:
            shifted_x = x
        # print ('shifted_x',shifted_x.shape)
        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size_T, self.window_size_E
        )  # nW*B, window_size_T, window_size_E, window_size_E, C
        # print ('x_windows',x_windows.shape)
        x_windows = x_windows.view(
            -1, self.window_size_T * self.window_size_E[0] * self.window_size_E[1], C
        )
        # nW*B, window_size_T*window_size_E*window_size_E, C
        # print ('x_windows',x_windows.shape)

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size_T, self.window_size_E[0], self.window_size_E[1], C
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size_T, self.window_size_E, T, H, W
        )  # B T' H' W' C
        # print ('attn_windows, shifted_x', attn_windows.shape, shifted_x.shape)

        # reverse cyclic shift
        if self.shift_size_E[0] > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size_T, self.shift_size_E[0], self.shift_size_E[0]),
                dims=(1, 2, 3),
            )
        else:
            x = shifted_x
        # print ('x shape', x.shape)
        x = x.view(B, T * H * W, C)
        # print ('x view',x.shape)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print ('out block', x.shape)
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size_T={self.window_size_T}, window_size_E={self.window_size_E}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):
    # Need to Change to 3D
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size_T,
        window_size_E,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        temporal_down=True,
        elec_down=True,
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size_T=window_size_T,
                    window_size_E=window_size_E,
                    shift_size_E=(0, 0)
                    if (i % 2 == 0)
                    else (window_size_E[0] // 2, window_size_E[1] // 2),
                    shift_size_T=0 if (i % 2 == 0) else window_size_T // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution,
                dim=dim,
                norm_layer=norm_layer,
                temporal_down=temporal_down,
                elec_down=elec_down,
            )
        else:
            self.downsample = None

    def forward(self, x):
        # print ('basic layer before blocks',x.shape)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        # print ('basic layer after blocks',x.shape)
        if self.downsample is not None:
            x = self.downsample(x)
        # print ('basic layer after downsample',x.shape)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class SwinTransformer(nn.Module):
    # No Need to Change to 3D
    r"""Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        temporal_size=128,
        img_size=16,
        patch_size_T=2,
        patch_size_E=2,
        in_chans=1,
        num_classes=64,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size_T=4,
        window_size_E=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.temporal_size = temporal_size
        self.img_size = img_size
        # split image into non-overlapping patches
        # temporal_size=128, img_size=16, patch_size_T=4, patch_size_E=4
        self.patch_embed = PatchEmbed(
            temporal_size=temporal_size,
            img_size=img_size,
            patch_size_T=patch_size_T,
            patch_size_E=patch_size_E,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        # print ('patches_resolution', patches_resolution)
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()

        # only downsample twice/three times
        downsample_time = self.num_layers - 1

        downsample_elec = [True for i_layer in range(self.num_layers)]
        if patch_size_E[0] >= 2:
            # if self.num_layers >=3:
            for i_layer in range(1, self.num_layers):
                downsample_elec[
                    i_layer
                ] = False  # which means we only downsample electrodes three times, then we only do temporal downsample if we have more depths
        else:
            # if self.num_layers >=4:
            for i_layer in range(2, self.num_layers):
                downsample_elec[
                    i_layer
                ] = False  # which means we only downsample electrodes three times, then we only do temporal downsample if we have more depths
        # E_downsample_times = np.prod([2**i for i in downsample_elec])

        for i_layer in range(self.num_layers):
            # print ('depths[:i_layer',depths, i_layer)
            # print ('sum',sum(depths[:i_layer]),sum(depths[:i_layer + 1]))
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1]
                    // int(
                        np.prod(
                            [2**downfact for downfact in downsample_elec[:i_layer]]
                        )
                    ),  # (2 ** i_layer),
                    patches_resolution[2]
                    // int(
                        np.prod(
                            [2**downfact for downfact in downsample_elec[:i_layer]]
                        )
                    ),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size_T=window_size_T,
                window_size_E=window_size_E,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    int(sum(depths[:i_layer])) : int(sum(depths[: i_layer + 1]))
                ],
                norm_layer=norm_layer,
                # downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                downsample=PatchMerging
                if (i_layer < downsample_time)
                else None,  # downsample only twice, if no patchembeding (patchsize=1), then 3 times
                use_checkpoint=use_checkpoint,
                temporal_down=True,  # temporal_down,
                elec_down=downsample_elec[i_layer],
            )
            self.layers.append(layer)
        # print ('E_down',int(np.prod([2**downfact for downfact in downsample_elec])))
        self.E_down_0 = (
            int(np.prod([2**downfact for downfact in downsample_elec]))
            * patch_size_E[0]
        )  # final downsample times for elec
        self.E_down_1 = (
            int(np.prod([2**downfact for downfact in downsample_elec]))
            * patch_size_E[1]
        )  # final downsample times for elec
        self.T_down = (
            2**downsample_time * patch_size_T
        )  # final downsample times for temporal
        # print ('E_down, T_down',self.E_down, self.T_down,np.prod([2**downfact for downfact in downsample_elec]),patch_size_E)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # print ('head feature dim',img_size//(2**len(depths))*embed_dim*((len(depths)-1)**2),img_size//(len(depths)**2), embed_dim,(2**(len(depths)-1)))
        # in_embed = img_size//(2**len(depths) )*img_size//(2**len(depths) ) *embed_dim*(2**(len(depths)-1))*(2**(len(depths)-1))
        in_embed = (
            img_size
            // (self.E_down_0)
            * img_size
            // (self.E_down_1)
            * embed_dim
            * (2 ** (downsample_time - 1))
            * (2 ** (downsample_time - 1))
            // 2 ** (self.num_layers - 3)
        )
        self.head = (
            nn.Linear(in_embed, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):
        # print ('before patch embed', x.shape)
        x = self.patch_embed(x)
        # print ('after patch embed', x.shape)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        # print (x.shape)
        for layer in self.layers:
            # print ('layer',layer)
            # print ('x shape before layer',x.shape)
            x = layer(x)
            # print ('x shape after layer',x.shape)
        # print (x.shape)
        x = self.norm(x)  # B L C
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # print (x.shape)
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        B, THW, C = x.shape
        T = self.temporal_size // self.T_down
        H = self.img_size // self.E_down_0
        W = self.img_size // self.E_down_1
        # print ('T, H', T, H)
        # print ('B, THW, C', B, THW, C)
        # original 128(16*8), 16, 16
        # H = int(HW**0.5)
        x = x.view(B, T, H, W, C)
        # print ('x, B, T, H, H, C',x.shape)
        x = x.view(B, T, H * W * C)
        # print ('xview', x.shape)
        # print ('head', self.head)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


class SwinTransformer_new(nn.Module):
    # No Need to Change to 3D
    r"""Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        temporal_size=128,
        img_size=16,
        patch_size_T=2,
        patch_size_E=2,
        in_chans=1,
        num_classes=64,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size_T=4,
        window_size_E=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.temporal_size = temporal_size
        self.img_size = img_size
        # split image into non-overlapping patches
        # temporal_size=128, img_size=16, patch_size_T=4, patch_size_E=4
        self.patch_embed = PatchEmbed(
            temporal_size=temporal_size,
            img_size=img_size,
            patch_size_T=patch_size_T,
            patch_size_E=patch_size_E,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        # print ('patches_resolution', patches_resolution)
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()

        # only downsample twice/three times
        # print ('self.num_layers',depths,self.num_layers)
        downsample_time = self.num_layers - 1
        # print ('downsample_time',downsample_time)
        downsample_elec = [True for i_layer in range(self.num_layers)]
        if patch_size_E[0] >= 2:
            # if self.num_layers >=3:
            for i_layer in range(0, self.num_layers):
                downsample_elec[
                    i_layer
                ] = False  # which means we only downsample electrodes three times, then we only do temporal downsample if we have more depths
        else:
            # if self.num_layers >=4:
            for i_layer in range(0, self.num_layers):
                downsample_elec[
                    i_layer
                ] = False  # which means we only downsample electrodes three times, then we only do temporal downsample if we have more depths
        # E_downsample_times = np.prod([2**i for i in downsample_elec])

        for i_layer in range(self.num_layers):
            # print ('depths[:i_layer',depths, i_layer)
            # print ('sum',sum(depths[:i_layer]),sum(depths[:i_layer + 1]))
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1]
                    // int(
                        np.prod(
                            [2**downfact for downfact in downsample_elec[:i_layer]]
                        )
                    ),  # (2 ** i_layer),
                    patches_resolution[2]
                    // int(
                        np.prod(
                            [2**downfact for downfact in downsample_elec[:i_layer]]
                        )
                    ),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size_T=window_size_T,
                window_size_E=window_size_E,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    int(sum(depths[:i_layer])) : int(sum(depths[: i_layer + 1]))
                ],
                norm_layer=norm_layer,
                # downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                downsample=PatchMerging
                if (i_layer < downsample_time)
                else None,  # downsample only twice, if no patchembeding (patchsize=1), then 3 times
                use_checkpoint=use_checkpoint,
                temporal_down=True,  # temporal_down,
                elec_down=downsample_elec[i_layer],
            )
            self.layers.append(layer)
        # print ('E_down',int(np.prod([2**downfact for downfact in downsample_elec])))
        self.E_down_0 = (
            int(np.prod([2**downfact for downfact in downsample_elec]))
            * patch_size_E[0]
        )  # final downsample times for elec
        self.E_down_1 = (
            int(np.prod([2**downfact for downfact in downsample_elec]))
            * patch_size_E[1]
        )  # final downsample times for elec
        self.T_down = (
            2**downsample_time * patch_size_T
        )  # final downsample times for temporal
        # print ('E_down, T_down',self.E_down, self.T_down,np.prod([2**downfact for downfact in downsample_elec]),patch_size_E)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # print ('head feature dim',img_size//(2**len(depths))*embed_dim*((len(depths)-1)**2),img_size//(len(depths)**2), embed_dim,(2**(len(depths)-1)))
        # in_embed = img_size//(2**len(depths) )*img_size//(2**len(depths) ) *embed_dim*(2**(len(depths)-1))*(2**(len(depths)-1))
        in_embed = (
            img_size
            // (self.E_down_0)
            * img_size
            // (self.E_down_1)
            * embed_dim
            * (2 ** (downsample_time - 1))
            * (2 ** (downsample_time - 1))
            // 2 ** (self.num_layers - 3)
        )
        self.head = (
            nn.Linear(in_embed, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):
        # print ('before patch embed', x.shape)
        x = self.patch_embed(x)
        # print ('after patch embed', x.shape)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        # print (x.shape)
        for layer in self.layers:
            # print ('layer',layer)
            # print ('x shape before layer',x.shape)
            x = layer(x)
            # print ('x shape after layer',x.shape)
        # print (x.shape)
        x = self.norm(x)  # B L C
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # print (x.shape)
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        B, THW, C = x.shape
        T = self.temporal_size // self.T_down
        H = self.img_size // self.E_down_0
        W = self.img_size // self.E_down_1
        # print ('T, H', T, H)
        # print ('B, THW, C', B, THW, C)
        # original 128(16*8), 16, 16
        # H = int(HW**0.5)
        x = x.view(B, T, H, W, C)
        # print ('x, B, T, H, H, C',x.shape)
        x = x.view(B, T, H * W * C)
        # print ('xview', x.shape)
        # print ('head', self.head)
        x = self.head(x)
        x = x.view(B, T, H, W, -1)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
import cv2 as cv
import numpy as np

from .feature_extractor import BasicBlock
from .warp import disp_warp


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class RefineAttention(nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type):
        super(RefineAttention, self).__init__()
        self.num_heads = num_heads
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x, mask_d, mask_e):
        b, c, h, w = x.shape
        x = self.norm(x)
        y1 = x * (1 - mask_e)
        y2 = x * (1 - mask_d)
        out_sa = x.clone()
        with torch.no_grad():
            for i in range(b):
                z_d = []
                z_e = []
                pos_d = np.argwhere(mask_d[i][0].cpu().detach().numpy() == 1)
                pos_e = np.argwhere(mask_e[i][0].cpu().detach().numpy() == 1)
                for j in range(c):
                    z_d.append(x[i, j, pos_d[:, 0], pos_d[:, 1]])
                    z_e.append(x[i, j, pos_e[:, 0], pos_e[:, 1]])
                # spatial attention
                z_d = torch.stack(z_d)
                z_e = torch.stack(z_e)
                z_e = z_e.cuda()
                z_d = z_d.cuda()
                k1 = rearrange(z_e, '(head c) z -> head z c', head=self.num_heads)
                v1 = rearrange(z_e, '(head c) z -> head z c', head=self.num_heads)
                q1 = rearrange(z_d, '(head c) z -> head z c', head=self.num_heads)
                q1 = torch.nn.functional.normalize(q1, dim=-1)
                k1 = torch.nn.functional.normalize(k1, dim=-1)
                attn1 = (q1 @ k1.transpose(-2, -1))
                attn1 = attn1.softmax(dim=-1)
                out1 = (attn1 @ v1) + q1
                out1 = rearrange(out1, 'head z c -> (head c) z', head=self.num_heads)
                for j in range(c):
                    out_sa[i, j, pos_d[:, 0], pos_d[:, 1]] = out1[j]
        # channel attention
        k2 = rearrange(y2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(y2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q2 = rearrange(y1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out2 = (attn2 @ v2) + q2
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = x + out_sa + out2

        return out


class StereoDRNetRefinement(nn.Module):
    def __init__(self, img_channels=3):
        super(StereoDRNetRefinement, self).__init__()

        # Left and warped error
        in_channels = img_channels * 2

        self.conv1 = conv2d(in_channels, 16)
        self.conv2 = conv2d(1, 16)  # on low disparity

        self.dilation_list = [1, 2, 4, 8, 1, 1]
        self.dilated_blocks = nn.ModuleList()

        for dilation in self.dilation_list:
            self.dilated_blocks.append(BasicBlock(32, 32, stride=1, dilation=dilation))

        self.dilated_blocks = nn.Sequential(*self.dilated_blocks)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

        # Adding RefineAttention
        self.refine_attention = RefineAttention(dim=1, num_heads=4, LayerNorm_type='WithBias')

    def forward(self, low_disp, left_img, right_img):
        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)  # [B, 1, H, W]
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        if scale_factor == 1.0:
            disp = low_disp
        else:
            disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
            disp = disp * scale_factor

        # Warp right image to left view with current disparity
        warped_right = disp_warp(right_img, disp)[0]  # [B, C, H, W]
        error = warped_right - left_img  # [B, C, H, W]

        concat1 = torch.cat((error, left_img), dim=1)  # [B, 6, H, W]

        conv1 = self.conv1(concat1)  # [B, 16, H, W]
        conv2 = self.conv2(disp)  # [B, 16, H, W]
        concat2 = torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W]

        out = self.dilated_blocks(concat2)  # [B, 32, H, W]
        residual_disp = self.final_conv(out)  # [B, 1, H, W]

        disp = F.relu(disp + residual_disp, inplace=True)  # [B, 1, H, W]

        # Apply RefineAttention
        disp = self.refine_attention(disp, mask_d=low_disp, mask_e=residual_disp)  # [B, 1, H, W]

        disp = disp.squeeze(1)  # [B, H, W]

        return disp


import torch

import torch.nn as nn

import torch.nn.functional as F

from timm.models.layers import DropPath

 

from .feature_extractor import BasicBlock

from .warp import disp_warp

 

 

def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):

    return nn.Sequential(

        nn.Conv2d(

            in_channels, out_channels, kernel_size=kernel_size,

            stride=stride, padding=dilation, dilation=dilation,

            bias=False, groups=groups

        ),

        nn.BatchNorm2d(out_channels),

        nn.LeakyReLU(0.2, inplace=True)

    )

 

 

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None,

                 act_layer=nn.GELU, drop=0.):

        super(Mlp, self).__init__()

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

 

 

class RoPE(torch.nn.Module):

    """旋转位置编码（Rotary Positional Embedding）。"""

    def __init__(self, shape, base=10000):

        super(RoPE, self).__init__()

 

        channel_dims, feature_dim = shape[:-1], shape[-1]

        k_max = feature_dim // (2 * len(channel_dims))

 

        assert feature_dim % k_max == 0

 

        # 角度计算

        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))

        mesh = torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')

        angles = torch.cat(

            [t.unsqueeze(-1) * theta_ks for t in mesh],

            dim=-1

        )

 

        # 旋转矩阵

        rotations_re = torch.cos(angles).unsqueeze(dim=-1)

        rotations_im = torch.sin(angles).unsqueeze(dim=-1)

        rotations = torch.cat([rotations_re, rotations_im], dim=-1)

        self.register_buffer('rotations', rotations)

 

    def forward(self, x):

        if x.dtype != torch.float32:

            x = x.to(torch.float32)

        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))

        rotations = self.rotations.to(x.device)  # 确保 rotations 在与 x 相同的设备上

        pe_x = torch.view_as_complex(rotations) * x

        return torch.view_as_real(pe_x).flatten(-2)

 

 

class LinearAttention(nn.Module):

    """线性注意力机制，包含 LePE 和 RoPE。"""

    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):

        super(LinearAttention, self).__init__()

        self.dim = dim

        self.num_heads = num_heads

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.elu = nn.ELU()

        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.rope = None  # 推迟 RoPE 的初始化

 

    def forward(self, x, h, w):

        b, n, c = x.shape

        num_heads = self.num_heads

        head_dim = c // num_heads

 

        # 在第一次前向传播时初始化 RoPE

        if self.rope is None:

            self.rope = RoPE(shape=(h, w, c)).to(x.device)

 

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)

        q, k, v = qk[0], qk[1], x

 

        q = self.elu(q) + 1.0

        k = self.elu(k) + 1.0

 

        # 应用 RoPE 编码

        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

 

        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

 

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)

        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))

        x = q_rope @ kv * z

 

        x = x.transpose(1, 2).reshape(b, n, c)

        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)

        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)

 

        return x

 

 

class MLLABlock(nn.Module):

    """MLLA 块，与源代码中的完全一致。"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,

                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):

        super(MLLABlock, self).__init__()

        self.dim = dim

        self.num_heads = num_heads

        self.mlp_ratio = mlp_ratio

 

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.norm1 = norm_layer(dim)

        self.in_proj = nn.Linear(dim, dim)

        self.act_proj = nn.Linear(dim, dim)

        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.act = nn.SiLU()

        self.attn = LinearAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.out_proj = nn.Linear(dim, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

 

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),

                       act_layer=act_layer, drop=drop)

 

    def forward(self, x, H, W):

        B, L, C = x.shape

        assert L == H * W, "输入特征尺寸不匹配"

 

        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)

        shortcut = x

 

        x = self.norm1(x)

        act_res = self.act(self.act_proj(x))

        x = self.in_proj(x).reshape(B, H, W, C)

        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).reshape(B, L, C)

 

        # 线性注意力

        x = self.attn(x, H, W)

 

        x = self.out_proj(x * act_res)

        x = shortcut + self.drop_path(x)

        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)

 

        # FFN

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

 

 

class StereoDRNetRefinement(nn.Module):

    def __init__(self, img_channels=3):

        super(StereoDRNetRefinement, self).__init__()

 

        # 左图像和误差输入通道

        in_channels = img_channels * 2

 

        self.conv1 = conv2d(in_channels, 16)

        self.conv2 = conv2d(1, 16)  # 对低分辨率的视差

 

        self.dilation_list = [1, 2, 4, 8, 1, 1]

        self.dilated_blocks = nn.ModuleList()

 

        for dilation in self.dilation_list:

            self.dilated_blocks.append(BasicBlock(32, 32, stride=1, dilation=dilation))

 

        self.dilated_blocks = nn.Sequential(*self.dilated_blocks)

 

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

 

        # 添加 MLLABlock

        self.mlla_dim = 32  # 设置 MLLABlock 的维度

        self.mlla_heads = 4  # 设置头数

        self.mlla_block = MLLABlock(dim=self.mlla_dim, num_heads=self.mlla_heads)

 

        # 用于将 disp 投影到指定的维度

        self.disp_proj = nn.Conv2d(1, self.mlla_dim, kernel_size=1)

 

        # 用于将 MLLABlock 输出还原为单通道

        self.disp_unproj = nn.Conv2d(self.mlla_dim, 1, kernel_size=1)

 

    def forward(self, low_disp, left_img, right_img):

        assert low_disp.dim() == 3

        low_disp = low_disp.unsqueeze(1)  # [B, 1, H, W]

        scale_factor = left_img.size(-1) / low_disp.size(-1)

        if scale_factor == 1.0:

            disp = low_disp

        else:

            disp = F.interpolate(

                low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False

            )

            disp = disp * scale_factor

 

        # Warp 右图像到左视图

        warped_right = disp_warp(right_img, disp)[0]  # [B, C, H, W]

        error = warped_right - left_img  # [B, C, H, W]

 

        concat1 = torch.cat((error, left_img), dim=1)  # [B, 6, H, W]

 

        conv1 = self.conv1(concat1)  # [B, 16, H, W]

        conv2 = self.conv2(disp)  # [B, 16, H, W]

        concat2 = torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W]

 

        out = self.dilated_blocks(concat2)  # [B, 32, H, W]

        residual_disp = self.final_conv(out)  # [B, 1, H, W]

 

        # 处理 disp，通过 MLLABlock

        B, _, H, W = disp.shape

 

        disp_feat = self.disp_proj(disp)  # [B, 32, H, W]

        disp_feat = disp_feat.permute(0, 2, 3, 1).reshape(B, H * W, self.mlla_dim)  # [B, L, C]

 

        disp_feat = self.mlla_block(disp_feat, H, W)  # [B, L, C]

        disp_feat = disp_feat.reshape(B, H, W, self.mlla_dim).permute(0, 3, 1, 2)  # [B, C, H, W]

 

        disp_processed = self.disp_unproj(disp_feat)  # [B, 1, H, W]

 

        # 将处理后的 disp 与 residual_disp 相加

        disp = F.relu(disp_processed + residual_disp, inplace=True)  # [B, 1, H, W]

        disp = disp.squeeze(1)  # [B, H, W]

 

        return disp
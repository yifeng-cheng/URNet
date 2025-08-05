import torch

import torch.nn as nn

import torch.nn.functional as F

 

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

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):

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

 

 

class LinearAttention(nn.Module):

    def __init__(self, dim):

        super(LinearAttention, self).__init__()

        self.qk = nn.Linear(dim, dim * 2, bias=False)

        self.elu = nn.ELU()

        self.eps = 1e-6  # 防止除零

 

    def forward(self, x):

        # x: [B, L, C]

        B, N, C = x.shape

 

        qk = self.qk(x).reshape(B, N, 2, C)

        q, k = qk[:, :, 0], qk[:, :, 1]

        q = self.elu(q) + 1

        k = self.elu(k) + 1

 

        # 线性注意力计算

        kv = torch.einsum('bnd,bne->bde', k, x)

        z = 1 / (torch.einsum('bnd,bd->bn', q, k.sum(dim=1)) + self.eps)

        x = torch.einsum('bnd,bde,bn->bne', q, kv, z)

 

        return x

 

 

class MLLABlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, drop=0.,

                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):

        super(MLLABlock, self).__init__()

        self.dim = dim

 

        # 卷积位置编码（CPE）

        self.cpe1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        self.cpe2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

 

        # 归一化层

        self.norm1 = norm_layer(dim)

        self.norm2 = norm_layer(dim)

 

        # 前馈网络的激活函数

        self.act = act_layer()

 

        # 前馈网络的线性层

        self.in_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.act_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias)

 

        # 深度卷积层

        self.dwc = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

 

        # 线性注意力

        self.attn = LinearAttention(dim)

 

        # DropPath

        self.drop_path = nn.Identity()  

 

        # MLP

        self.mlp = Mlp(

            in_features=dim,

            hidden_features=int(dim * mlp_ratio),

            act_layer=act_layer,

            drop=drop

        )

 

    def forward(self, x, H, W):

        B, L, C = x.shape

        assert L == H * W, "输入特征的尺寸不匹配"

 

        # 卷积位置编码

        x = x + self.cpe1(x.transpose(1, 2).view(B, C, H, W)).view(B, C, L).transpose(1, 2)

        shortcut = x

 

        x = self.norm1(x)

        act_res = self.act(self.act_proj(x))

        x = self.in_proj(x).transpose(1, 2).view(B, C, H, W)

        x = self.act(self.dwc(x)).view(B, C, L).transpose(1, 2)

 

        # 线性注意力

        x = self.attn(x)

 

        x = self.out_proj(x * act_res)

        x = shortcut + self.drop_path(x)

 

        # 第二次卷积位置编码

        x = x + self.cpe2(x.transpose(1, 2).view(B, C, H, W)).view(B, C, L).transpose(1, 2)

 

        # MLP

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

        self.mlla_block = MLLABlock(dim=self.mlla_dim)

 

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

        disp_feat = disp_feat.view(B, self.mlla_dim, H * W).transpose(1, 2)  # [B, H*W, 32]

 

        disp_feat = self.mlla_block(disp_feat, H, W)  # [B, H*W, 32]

        disp_feat = disp_feat.transpose(1, 2).view(B, self.mlla_dim, H, W)  # [B, 32, H, W]

 

        disp_processed = self.disp_unproj(disp_feat)  # [B, 1, H, W]

 

        # 将处理后的 disp 与 residual_disp 相加

        disp = F.relu(disp_processed + residual_disp, inplace=True)  # [B, 1, H, W]

        disp = disp.squeeze(1)  # [B, H, W]

 

        return disp
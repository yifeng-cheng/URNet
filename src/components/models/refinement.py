import torch
import torch.nn as nn
import torch.nn.functional as F
from .feature_extractor import BasicBlock
from .warp import disp_warp

def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=False, groups=groups), nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True))

class RoPE(torch.nn.Module):
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()
        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))
        assert feature_dim % k_max == 0
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        mesh = torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in mesh], dim=-1)
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)
     
    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        rotations = self.rotations.to(x.device)
        pe_x = torch.view_as_complex(rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)

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
    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):
        super(LinearAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = None
     
    def forward(self, x, h, w):
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads
     
        if self.rope is None:
            self.rope = RoPE(shape=(h, w, c)).to(x.device)
         
        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k = qk[0], qk[1]
        v = x
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
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
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
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
        self.drop_path = nn.Identity()
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
     
    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)
        shortcut = x
        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).reshape(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).reshape(B, L, C)
        x = self.attn(x, H, W)
        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
     
        return x

class RefUnet(nn.Module):
    def __init__(self, in_ch, inc_ch):
        super(RefUnet, self).__init__()
        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)
        self.conv1 = nn.Conv2d(inc_ch, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=False)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=False)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=False)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=False)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)
        self.mlla_dim = 64
        self.mlla_heads = 4
        self.mlla_block = MLLABlock(dim=self.mlla_dim, num_heads=self.mlla_heads)
        self.proj_in = nn.Conv2d(64, self.mlla_dim, kernel_size=1)
        self.proj_out = nn.Conv2d(self.mlla_dim, 64, kernel_size=1)
        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)
        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)
        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)
        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)
        self.conv_d0 = nn.Conv2d(64, 1, 3, padding=1)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
     
    def forward(self, x):
        hx = x
        hx = self.conv0(hx)
        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)
        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)
        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)
        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)
        hx5 = self.relu5(self.bn5(self.conv5(hx)))
        B, C, H, W = hx5.shape
        x_mlla = self.proj_in(hx5)
        x_mlla = x_mlla.permute(0, 2, 3, 1).reshape(B, H * W, self.mlla_dim)
        x_mlla = self.mlla_block(x_mlla, H, W)
        x_mlla = x_mlla.reshape(B, H, W, self.mlla_dim).permute(0, 3, 1, 2)
        x_mlla = self.proj_out(x_mlla)
        hx5 = hx5 + x_mlla
        hx = self.upscore2(hx5)
        hx = F.interpolate(hx, size=hx4.size()[2:], mode='bilinear', align_corners=False)
        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))
        hx = self.upscore2(d4)
        hx = F.interpolate(hx, size=hx3.size()[2:], mode='bilinear', align_corners=False)
        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))
        hx = self.upscore2(d3)
        hx = F.interpolate(hx, size=hx2.size()[2:], mode='bilinear', align_corners=False)
        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))
        hx = self.upscore2(d2)
        hx = F.interpolate(hx, size=hx1.size()[2:], mode='bilinear', align_corners=False)
        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))
        residual = self.conv_d0(d1)
     
        return x + residual

class StereoDRNetRefinement(nn.Module):
    def __init__(self, img_channels=3):
        super(StereoDRNetRefinement, self).__init__()
        in_channels = img_channels * 2
        self.conv1 = conv2d(in_channels, 16)
        self.conv2 = conv2d(1, 16)
        self.dilation_list = [1, 2, 4, 8, 1, 1]
        self.dilated_blocks = nn.ModuleList()
        for dilation in self.dilation_list:
            self.dilated_blocks.append(BasicBlock(32, 32, stride=1, dilation=dilation))
        self.dilated_blocks = nn.Sequential(*self.dilated_blocks)
        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)
        self.refunet = RefUnet(in_ch=1, inc_ch=16)
     
    def forward(self, low_disp, left_img, right_img):
        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)
        scale_factor = left_img.size(-1) / low_disp.size(-1)
     
        if scale_factor == 1.0:
            disp = low_disp
        else:
            disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
            disp = disp * scale_factor
         
        warped_right = disp_warp(right_img, disp)[0]
        error = warped_right - left_img
        concat1 = torch.cat((error, left_img), dim=1)
        conv1 = self.conv1(concat1)
        conv2 = self.conv2(disp)
        concat2 = torch.cat((conv1, conv2), dim=1)
        out = self.dilated_blocks(concat2)
        residual_disp = self.final_conv(out)
        refined_disp = self.refunet(disp)
        refined_disp = F.relu(refined_disp + residual_disp, inplace=True)
        refined_disp = refined_disp.squeeze(1)
     
        return refined_disp

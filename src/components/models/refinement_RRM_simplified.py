import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import BasicBlock
from .warp import disp_warp


# 定义卷积层的辅助函数
def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))


class RefUnet(nn.Module):
    def __init__(self, in_ch, inc_ch):
        super(RefUnet, self).__init__()

        # 初始卷积层
        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)

        # 第一层下采样
        self.conv1 = nn.Conv2d(inc_ch, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=False)

        # 第二层下采样
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=False)

        # 第二层上采样
        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        # 第一层上采样
        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        # 最终卷积层
        self.conv_d0 = nn.Conv2d(64, 1, 3, padding=1)

        # 上采样操作
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        hx = x
        hx = self.conv0(hx)

        # 第一层下采样
        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        # 第二层下采样
        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        # 第二层上采样
        hx = self.upscore2(hx)  # 上采样
        hx = F.interpolate(hx, size=hx2.size()[2:], mode='bilinear', align_corners=True)
        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))

        # 第一层上采样
        hx = self.upscore2(d2)  # 上采样
        hx = F.interpolate(hx, size=hx1.size()[2:], mode='bilinear', align_corners=True)
        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))

        # 最终卷积层
        residual = self.conv_d0(d1)

        return x + residual


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

        # Initialize the RefUnet
        self.refunet = RefUnet(in_ch=1, inc_ch=16)

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

        # Use RefUnet to refine the disp
        refined_disp = self.refunet(disp)  # [B, 1, H, W]
        
        refined_disp = F.relu(refined_disp + residual_disp, inplace=True)  # [B, 1, H, W]

        refined_disp = refined_disp.squeeze(1)  # [B, H, W]

        return refined_disp

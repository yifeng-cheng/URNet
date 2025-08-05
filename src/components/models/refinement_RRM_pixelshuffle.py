import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import BasicBlock
from .warp import disp_warp


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=dilation, dilation=dilation,
                  bias=False, groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )


class RefUnet(nn.Module):
    def __init__(self, in_ch, inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)

        self.conv1 = nn.Conv2d(inc_ch, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        # Pixel Shuffle部分
        self.conv_ps5 = nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1)
        self.ps5 = nn.PixelShuffle(2)

        self.conv_ps4 = nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1)
        self.ps4 = nn.PixelShuffle(2)

        self.conv_ps3 = nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1)
        self.ps3 = nn.PixelShuffle(2)

        self.conv_ps2 = nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1)
        self.ps2 = nn.PixelShuffle(2)

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

        # 第一次上采样
        hx = self.conv_ps5(hx5)
        hx = self.ps5(hx)

        if hx.size() != hx4.size():
            hx = F.interpolate(hx, size=hx4.size()[2:], mode='bilinear', align_corners=True)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))

        # 第二次上采样
        hx = self.conv_ps4(d4)
        hx = self.ps4(hx)

        if hx.size() != hx3.size():
            hx = F.interpolate(hx, size=hx3.size()[2:], mode='bilinear', align_corners=True)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))

        # 第三次上采样
        hx = self.conv_ps3(d3)
        hx = self.ps3(hx)

        if hx.size() != hx2.size():
            hx = F.interpolate(hx, size=hx2.size()[2:], mode='bilinear', align_corners=True)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))

        # 第四次上采样
        hx = self.conv_ps2(d2)
        hx = self.ps2(hx)

        if hx.size() != hx1.size():
            hx = F.interpolate(hx, size=hx1.size()[2:], mode='bilinear', align_corners=True)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))

        residual = self.conv_d0(d1)

        return x + residual


class StereoDRNetRefinement(nn.Module):
    def __init__(self, img_channels=3):
        super(StereoDRNetRefinement, self).__init__()

        # 左图和误差
        in_channels = img_channels * 2

        self.conv1 = conv2d(in_channels, 16)
        self.conv2 = conv2d(1, 16)  # 处理低分辨率视差

        self.dilation_list = [1, 2, 4, 8, 1, 1]
        self.dilated_blocks = nn.ModuleList()

        for dilation in self.dilation_list:
            self.dilated_blocks.append(BasicBlock(32, 32, stride=1, dilation=dilation))

        self.dilated_blocks = nn.Sequential(*self.dilated_blocks)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

        # 初始化 RefUnet
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

        print(f"Minimum disparity value before warp: {disp.min().item()}")
        # 使用当前视差将右图向左图视角进行warp
        warped_right = disp_warp(right_img, disp)[0]  # [B, C, H, W]
        error = warped_right - left_img  # [B, C, H, W]

        concat1 = torch.cat((error, left_img), dim=1)  # [B, 6, H, W]

        conv1 = self.conv1(concat1)  # [B, 16, H, W]
        conv2 = self.conv2(disp)  # [B, 16, H, W]
        concat2 = torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W]

        out = self.dilated_blocks(concat2)  # [B, 32, H, W]
        residual_disp = self.final_conv(out)  # [B, 1, H, W]

        # 使用 RefUnet 对 disp 进行精炼
        refined_disp = self.refunet(disp)  # [B, 1, H, W]

        refined_disp = F.relu(refined_disp + residual_disp, inplace=True)  # [B, 1, H, W]
        
        refined_disp = refined_disp.squeeze(1)  # [B, H, W]

        return refined_disp

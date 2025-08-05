import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch

# 导入自适应损失函数
from .adaptive import AdaptiveImageLossFunction
from .concentration import ConcentrationNet
from .stereo_matching import StereoMatchingNetwork

class EventStereoMatchingNetwork(nn.Module):
    def __init__(self, 
                 concentration_net=None,
                 disparity_estimator=None):
        super(EventStereoMatchingNetwork, self).__init__()
        self.concentration_net = ConcentrationNet(**concentration_net.PARAMS)
        self.stereo_matching_net = StereoMatchingNetwork(**disparity_estimator.PARAMS)

        # 延迟初始化自适应损失函数
        self.criterion = None  # 初始设置为 None

    def forward(self, left_event, right_event, gt_disparity=None):
        event_stack = {
            'l': left_event.clone(),
            'r': right_event.clone(),
        }
        concentrated_event_stack = {}
        for loc in ['l', 'r']:
            # 重排列事件堆栈
            event_stack[loc] = rearrange(event_stack[loc], 'b c h w t s -> b (c s t) h w')
            concentrated_event_stack[loc] = self.concentration_net(event_stack[loc])

        # 预测视差金字塔
        pred_disparity_pyramid = self.stereo_matching_net(
            concentrated_event_stack['l'],
            concentrated_event_stack['r']
        )

        loss_disp = None
        if gt_disparity is not None:
            # 计算损失
            loss_disp = self._cal_loss(pred_disparity_pyramid, gt_disparity)

        return pred_disparity_pyramid[-1], loss_disp

    def get_params_group(self, learning_rate):
        def filter_specific_params(kv):
            specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
            for name in specific_layer_name:
                if name in kv[0]:
                    return True
            return False

        def filter_base_params(kv):
            specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
            for name in specific_layer_name:
                if name in kv[0]:
                    return False
            return True

        specific_params = list(filter(filter_specific_params,
                                      self.named_parameters()))
        base_params = list(filter(filter_base_params,
                                  self.named_parameters()))

        specific_params = [kv[1] for kv in specific_params]  # kv是键值对 (key, value)
        base_params = [kv[1] for kv in base_params]

        specific_lr = learning_rate * 0.1
        params_group = [
            {'params': base_params, 'lr': learning_rate},
            {'params': specific_params, 'lr': specific_lr},
        ]

        return params_group

    def _cal_loss(self, pred_disparity_pyramid, gt_disparity):
        # 金字塔的权重
        pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]

        loss = 0.0
        mask = gt_disparity > 0  # 只计算视差大于0的部分
        for idx in range(len(pred_disparity_pyramid)):
            pred_disp = pred_disparity_pyramid[idx]
            weight = pyramid_weight[idx]

            # 如果预测的视差尺寸与gt视差不一致，进行插值
            if pred_disp.size(-1) != gt_disparity.size(-1):
                pred_disp = pred_disp.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
                pred_disp = F.interpolate(pred_disp, size=(gt_disparity.size(-2), gt_disparity.size(-1)),
                                          mode='bilinear', align_corners=False) * (
                                        gt_disparity.size(-1) / pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)  # [B, 1, H, W] -> [B, H, W]

            # 计算残差
            residuals = pred_disp - gt_disparity  # [B, H, W]

            # 添加通道维度
            residuals = residuals.unsqueeze(-1)  # [B, H, W] -> [B, H, W, 1]

            # 应用掩码
            mask_expanded = mask.unsqueeze(-1)  # [B, H, W] -> [B, H, W, 1]
            residuals = residuals * mask_expanded

            # 动态初始化损失函数
            if self.criterion is None:
                batch_size, height, width, channels = residuals.shape
                self.criterion = AdaptiveImageLossFunction(
                    image_size=(height, width, channels),
                    float_dtype=residuals.dtype,
                    device=residuals.device,
                    color_space='RGB',
                    representation='PIXEL',
                    use_students_t=False
                )

            # 计算损失
            cur_loss = self.criterion.lossfun(residuals)  # [B, H, W, 1]

            # 计算有效像素的数量
            valid_pixel_count = mask.sum()

            # 累加损失
            if valid_pixel_count > 0:
                loss += weight * (cur_loss.sum() / valid_pixel_count)
            else:
                loss += 0.0  # 如果没有有效像素，损失为零

        return loss

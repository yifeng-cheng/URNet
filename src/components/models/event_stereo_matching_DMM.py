import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .concentration import ConcentrationNet
from .stereo_matching import StereoMatchingNetwork


class EventStereoMatchingNetwork(nn.Module):
    def __init__(self, concentration_net=None, disparity_estimator=None):
        super(EventStereoMatchingNetwork, self).__init__()
        self.concentration_net = ConcentrationNet(**concentration_net.PARAMS)
        self.stereo_matching_net = StereoMatchingNetwork(**disparity_estimator.PARAMS)

    def forward(self, left_event, right_event, gt_disparity=None):
        event_stack = {
            'l': left_event.clone(),
            'r': right_event.clone(),
        }
        concentrated_event_stack = {}
        for loc in ['l', 'r']:
            event_stack[loc] = rearrange(event_stack[loc], 'b c h w t s -> b (c s t) h w')
            concentrated_event_stack[loc] = self.concentration_net(event_stack[loc])

        # stereo_matching_net returns a list of tuples (pred_disp, pred_log_var)
        pred_disparity_pyramid = self.stereo_matching_net(
            concentrated_event_stack['l'],
            concentrated_event_stack['r']
        )

        loss_disp = None
        if gt_disparity is not None:
            loss_disp = self._cal_loss(pred_disparity_pyramid, gt_disparity)

        # Return the last predicted disparity and log variance
        pred_disp, pred_log_var = pred_disparity_pyramid[-1]
        return pred_disp, pred_log_var, loss_disp

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

        specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
        base_params = [kv[1] for kv in base_params]

        specific_lr = learning_rate * 0.1
        params_group = [
            {'params': base_params, 'lr': learning_rate},
            {'params': specific_params, 'lr': specific_lr},
        ]

        return params_group

    def _cal_loss(self, pred_disparity_pyramid, gt_disparity):
        pyramid_weight = [1 / 3, 2 / 3, 1.0] + [1.0] * (len(pred_disparity_pyramid) - 3)

        loss = 0.0
        mask = gt_disparity > 0  # 有效视差掩码
        for idx in range(len(pred_disparity_pyramid)):
            pred_disp, pred_log_var = pred_disparity_pyramid[idx]
            weight = pyramid_weight[idx]

            # 如果需要，调整预测的尺寸以匹配 ground truth
            if pred_disp.size(-1) != gt_disparity.size(-1) or pred_disp.size(-2) != gt_disparity.size(-2):
                scale_factor_h = gt_disparity.size(-2) / pred_disp.size(-2)
                scale_factor_w = gt_disparity.size(-1) / pred_disp.size(-1)
                
                # 调整 pred_disp 尺寸
                pred_disp = F.interpolate(pred_disp.unsqueeze(1),
                                          size=(gt_disparity.size(-2), gt_disparity.size(-1)),
                                          mode='bilinear', align_corners=False) * scale_factor_w
                pred_disp = pred_disp.squeeze(1)

            # 调整 pred_log_var 尺寸
            if pred_log_var is not None:
                pred_log_var = F.interpolate(pred_log_var.unsqueeze(1),
                                             size=(gt_disparity.size(-2), gt_disparity.size(-1)),
                                             mode='bilinear', align_corners=False)
                pred_log_var = pred_log_var.squeeze(1)

            valid_mask = mask
            if valid_mask.sum() == 0:
                continue  # 如果没有有效像素，跳过

            # 计算损失
            delta = pred_disp[valid_mask] - gt_disparity[valid_mask]
            regression_loss = F.smooth_l1_loss(pred_disp[valid_mask],
                                               gt_disparity[valid_mask],
                                               reduction='mean')

            if pred_log_var is not None:
                log_var = pred_log_var[valid_mask]
                # 对预测的对数方差进行裁剪，防止数值问题
                log_var = torch.clamp(log_var, min=-10, max=10)
                var = torch.exp(log_var)

                # 二阶矩匹配项
                second_moment_matching = F.smooth_l1_loss(var,
                                                          delta ** 2,
                                                          reduction='mean')

                # 该尺度的总损失
                loss_i = regression_loss + second_moment_matching
            else:
                # 仅计算回归损失
                loss_i = regression_loss

            loss += weight * loss_i

        return loss

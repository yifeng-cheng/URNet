import torch
import torch.nn as nn
import torch.nn.functional as F

class DisparityEstimation(nn.Module):
    def __init__(self, max_disp, match_similarity=True):
        super(DisparityEstimation, self).__init__()

        self.max_disp = max_disp
        self.match_similarity = match_similarity

    def forward(self, cost_volume):
        assert cost_volume.dim() == 4  # [B, D, H, W]

        cost_volume = cost_volume if self.match_similarity else -cost_volume

        prob_volume = F.softmax(cost_volume, dim=1)  # [B, D, H, W]

        D = cost_volume.size(1)
        disp_candidates = torch.arange(0, D, device=cost_volume.device).float()
        disp_candidates = disp_candidates.view(1, D, 1, 1)  # [1, D, 1, 1]

        mean_disp = torch.sum(prob_volume * disp_candidates, dim=1)  # [B, H, W]

        disp_variance = torch.sum(
            prob_volume * (disp_candidates - mean_disp.unsqueeze(1)) ** 2, dim=1
        )  # [B, H, W]

        log_variance = torch.log(disp_variance + 1e-6)  

        return mean_disp, log_variance


class DisparityEstimationPyramid(nn.Module):
    def __init__(self, max_disp, match_similarity=True):
        super(DisparityEstimationPyramid, self).__init__()
        self.disparity_estimator = DisparityEstimation(max_disp, match_similarity)

    def forward(self, cost_volume_pyramid):
        disparity_pyramid = []
        for cost_volume in cost_volume_pyramid:
            mean_disp, log_variance = self.disparity_estimator(cost_volume)
            disparity_pyramid.append((mean_disp, log_variance))

        return disparity_pyramid[::-1]  

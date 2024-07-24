import torch
import torch.nn as nn

from frame_field_learning import frame_field_utils
from ..builder import LOSSES




@LOSSES.register_module()
class CrossfieldSmoothLoss(nn.Module):
    def __init__(self):
        super(CrossfieldSmoothLoss, self).__init__()
        self.laplacian_penalty = frame_field_utils.LaplacianPenalty(channels=4)

    def forward(self, pre_crossfield):
        penalty = self.laplacian_penalty(pre_crossfield)
        # avg_penalty = torch.mean(penalty * gt_edges_inv[:, None, ...])
        avg_penalty = torch.mean(penalty)# no mask
        return avg_penalty

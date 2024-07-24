import torch
import torch.nn as nn
from ..builder import LOSSES
import torch.nn.functional as F



@LOSSES.register_module()
class HorizontalSegGradLoss(nn.Module):
    def __init__(self, left_channel,right_channel):
        super(HorizontalSegGradLoss, self).__init__()
        self.left_channel = left_channel
        self.right_channel = right_channel

    def forward(self, pred_seg):
        seg_interior_roof_x_grad = pred_seg['grads'][:,self.right_channel,1,...]
        seg_interior_facade_x_grad = pred_seg['grads'][:,self.left_channel,1,...]
        return torch.abs(F.relu(seg_interior_roof_x_grad) - F.relu(-seg_interior_facade_x_grad))
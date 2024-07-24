import torch
import torch.nn as nn
from frame_field_learning import frame_field_utils
from torch_lydorn.torch.utils.complex import complex_abs_squared
from ..builder import LOSSES


@LOSSES.register_module()
class CrossfieldAlignOffNadirLoss(nn.Module):
    def __init__(self,level_2_align=False):
        super(CrossfieldAlignOffNadirLoss, self).__init__()
        self.level_2_align=level_2_align

    def forward(self, pred_c,gt_field,gt_masks=None): # mask
        c0 = pred_c[:, :2]
        c2 = pred_c[:, 2:]
        assert 2 <= gt_masks.shape[1], \
            "gt_polygons_image should have at least 2 channels for interior and edges"
        if self.level_2_align:
            align_loss = frame_field_utils.framefield_align_error_2level(c0, c2, gt_field, complex_dim=1)
        else:
            align_loss = frame_field_utils.framefield_align_error(c0, c2, gt_field, complex_dim=1)
        mask=torch.logical_or(torch.logical_or(gt_masks[...,1]==1,gt_masks[...,1]==2),
                              gt_masks[...,0]==1)
        avg_align_loss = torch.mean(align_loss * mask)
        # avg_align_loss = torch.mean(align_loss)
        c0_abs_loss=torch.mean(torch.abs(1-complex_abs_squared(c0,complex_dim=1)))
        c2_abs_loss=torch.mean(torch.abs(1-complex_abs_squared(c2,complex_dim=1)))
        return avg_align_loss+c0_abs_loss+c2_abs_loss
        # (gt_field[:,0] [mask]).reshape((2,-1)).mean(axis=-1)
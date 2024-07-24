import torch
import torch.nn as nn
from frame_field_learning import frame_field_utils
from ..builder import LOSSES


@LOSSES.register_module()
class CrossfieldGradAlignLoss(nn.Module):
    def __init__(self,pre_channel,level_2_align=False):
        super(CrossfieldGradAlignLoss, self).__init__()
        self.pre_channel=pre_channel if isinstance(pre_channel,list) else [pre_channel]
        self.level_2_align=level_2_align

    def forward(self,mask_grad,crossfield_targets,reduction='mean'):
        # TODO: don't apply on corners: corner_map = gt_batch["gt_polygons_image"][:, 2, :, :]
        # TODO: apply on all seg at once? Like seg is now?
        c0 = crossfield_targets[:, :2]
        c2 = crossfield_targets[:, 2:]
        result_align_loss=0
        for channel in self.pre_channel:
            seg_slice_grads_normed = mask_grad["grads_normed"][:, channel, ...]
            seg_slice_grad_norm = mask_grad["grad_norm"][:, channel, ...]
            if self.level_2_align:
                align_loss = frame_field_utils.framefield_align_error_2level(c0, c2, seg_slice_grads_normed, complex_dim=1)
            else:
                align_loss = frame_field_utils.framefield_align_error(c0, c2, seg_slice_grads_normed, complex_dim=1)
            # normed_align_loss = align_loss * seg_slice_grad_norm
            # avg_align_loss = torch.sum(normed_align_loss) / (torch.sum(seg_slice_grad_norm) + 1e-6)
            if reduction=='mean':
                result_align_loss += torch.mean(align_loss * seg_slice_grad_norm.detach())
            elif reduction=='sum':
                result_align_loss += torch.sum(align_loss * seg_slice_grad_norm.detach())
            # (prev line) Don't back-propagate to seg_slice_grad_norm so that seg smoothness is not encouraged
            # Save extra info for viz:
            # self.extra_info["seg_slice_grads"] = mask_grad["grads"][:, channel, ...]
        return result_align_loss
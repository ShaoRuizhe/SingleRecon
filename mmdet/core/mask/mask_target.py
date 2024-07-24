import numpy as np
import torch
from mmcv.ops import roi_align
from torch.nn.functional import one_hot
from torch.nn.modules.utils import _pair

from .structures import BitmapMasks


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg,instance_indicator=None):
    """Compute mask target for positive proposals in multiple images.

    Args:
        pos_proposals_list (list[Tensor]): Positive proposals in multiple
            images.
        pos_assigned_gt_inds_list (list[Tensor]): Assigned GT indices for each
            positive proposals.
        gt_masks_list (list[:obj:`BaseInstanceMasks`]): Ground truth masks of
            each image.
        cfg (dict): Config dict that specifies the mask size.

    Returns:
        list[Tensor]: Mask target of each image.
    """
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    if instance_indicator is None:
        instance_indicator=[None]*len(pos_proposals_list)
    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list,instance_indicator)
    mask_targets = [mask_target for mask_target in mask_targets if len(mask_target)>0]
    if len(mask_targets) > 0:
        mask_targets = torch.cat(mask_targets)
    return mask_targets


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg,instance_indicator=None):
    """Compute mask target for each positive proposal in the image.

    Args:
        pos_proposals (Tensor): Positive proposals.
        pos_assigned_gt_inds (Tensor): Assigned GT inds of positive proposals.
        gt_masks (:obj:`BaseInstanceMasks`): GT masks in the format of Bitmap
            or Polygon.
        cfg (dict): Config dict that indicate the mask size.

    Returns:
        Tensor: Mask target of each positive proposals in the image.
    """
    device = pos_proposals.device
    mask_size = _pair(cfg.mask_size)
    num_pos = pos_proposals.size(0)
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        if isinstance(gt_masks,BitmapMasks):
            maxh, maxw = gt_masks.height, gt_masks.width
        elif isinstance(gt_masks,torch.Tensor):
            maxh, maxw =(1024,1024)
        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        if isinstance(gt_masks, BitmapMasks):
            mask_targets = gt_masks.crop_and_resize(
                proposals_np, mask_size, device=device,
                inds=pos_assigned_gt_inds).to_ndarray()
            mask_targets = torch.from_numpy(mask_targets).float().to(device)
        elif isinstance(gt_masks,torch.Tensor):
            if instance_indicator is not None:
                indicator_masks = (
                            instance_indicator[:, :, None].repeat(1, 1, len(pos_assigned_gt_inds)) == torch.tensor(
                        pos_assigned_gt_inds).to(device))
                gt_masks_th = indicator_masks * gt_masks.permute(2, 0, 1)[..., None]# borcast:
                gt_masks_th=gt_masks_th.to(torch.int64).permute(0,3,1,2)
            if len(gt_masks_th.shape)==4:
                gt_masks_interior_th=one_hot(gt_masks_th[0],3).permute(0,3,1,2)# background,roof,facade 3 classses
                # gt_masks_seg_th=one_hot(gt_masks_th[1],4).permute(0,3,1,2)# background,roof_right,between,facade_up_down 3 classses
                # gt_masks_vertex_th=one_hot(gt_masks_th[2],2).permute(0,3,1,2)# background,vertex 2 classses
            proposals_tensor = torch.tensor(proposals_np, device=device)
            fake_inds = torch.arange(len(proposals_np)).to(device=device,dtype=proposals_tensor.dtype)[:, None]
            proposals_tensor = torch.cat([fake_inds, proposals_tensor], dim=1)
            targets = roi_align(gt_masks_interior_th.float().contiguous(), proposals_tensor, mask_size,
                                gt_masks_interior_th.shape[2]/1024, 0, 'avg', True).squeeze(1)# 似乎是不能多个channle一起？？不然会变成条纹  也不能用gpu...
            mask_targets = (targets >= 0.5)
    else:
        mask_targets = pos_proposals.new_zeros((0, ) + mask_size)
    return mask_targets

# encoding:utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_upsample_layer
from mmcv.ops import Conv2d
from mmcv.ops.carafe import CARAFEPack
from torch.nn.modules.utils import _pair

import torch_lydorn
from mmdet.core import force_fp32, mask_target
from ...builder import HEADS, build_loss,build_roi_extractor
from .fcn_mask_head import FCNMaskHead
from mmcv import ops
BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit

class ComputeSegGrads(torch.nn.Module):
    def __init__(self, device):
        super(ComputeSegGrads, self).__init__()
        self.tensor=torch.ones(23)
        self.spatial_gradient = torch_lydorn.kornia.filters.SpatialGradient(mode="scharr", coord="ij", normalized=True, device=device)

    def forward(self, pred_batch):
        seg = pred_batch  # (b, c, h, w)
        seg_grads = 2 * self.spatial_gradient(seg)  # (b, c, 2, h, w), Normalize (kornia normalizes to -0.5, 0.5 for input in [0, 1])
        seg_grad_norm = seg_grads.norm(dim=2)  # (b, c, h, w)
        seg_grads_normed = seg_grads / (seg_grad_norm[:, :, None, ...] + 1e-6)  # (b, c, 2, h, w)
        mask_grads={}
        mask_grads["grads"] = seg_grads
        mask_grads["grad_norm"] = seg_grad_norm
        mask_grads["grads_normed"] = seg_grads_normed
        return mask_grads

@HEADS.register_module()
class FCNMaskCrossfieldConstraintHead(FCNMaskHead):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=80,
                 class_agnostic=False,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 conv_cfg=None,
                 norm_cfg=None,
                 crossfield_roi_align=dict(type='RoIAlign', output_size=28, sampling_ratio=0),
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 mask_channel=None,
                 loss_roof_crossfield=None,
                 loss_roof_facade_horiontal=None,
                 ):
        """

        Args:
            num_convs:
            roi_feat_size:
            in_channels:
            conv_kernel_size:
            conv_out_channels:
            num_classes:
            class_agnostic:
            upsample_cfg:
            conv_cfg:
            norm_cfg:
            crossfield_roi_align:
            loss_mask:
            mask_channel: 允许哪些channel加入mask损失计算
            loss_roof_crossfield:
            loss_roof_facade_horiontal:
        """
        super(FCNMaskCrossfieldConstraintHead, self).__init__(num_convs,
                 roi_feat_size,
                 in_channels,
                 conv_kernel_size,
                 conv_out_channels,
                 num_classes,
                 class_agnostic,
                 upsample_cfg,
                 conv_cfg,
                 norm_cfg,
                 loss_mask)
        self.loss_roof_crossfield=build_loss(loss_roof_crossfield)if loss_roof_crossfield is not None else None
        self.mask_channel=mask_channel
        self.loss_roof_facade_horiontal=build_loss(loss_roof_facade_horiontal)if loss_roof_facade_horiontal is not None else None
        self.grad_calculator = ComputeSegGrads(device=torch.device('cuda'))# 使用torch.cuda.set_device(int_X/'cuda:X')之后，torch.device('cuda')会自动获取到前面指定的卡
        self.crossfield_roi_align=self.build_roi_layer(crossfield_roi_align)

    def build_roi_layer(self, layer_cfg):
        """Build RoI operator to extract feature from each level feature map.

        Args:
            layer_cfg (dict): Dictionary to construct and config RoI layer
                operation. Options are modules under ``mmcv/ops`` such as
                ``RoIAlign``.
            featmap_strides (int): The stride of input feature map w.r.t to the
                original image size, which would be used to scale RoI
                coordinate (original image coordinate system) to feature
                coordinate system.

        Returns:
            nn.ModuleList: The RoI extractor modules for each level feature
                map.
        """

        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        return layer_cls(**cfg)

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum() * 0
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_pred, mask_targets,
                                           torch.zeros_like(labels))
            else:
                if self.mask_channel is not None:
                    loss_mask = self.loss_mask(mask_pred[:,self.mask_channel], mask_targets[:,self.mask_channel], labels)
                else:
                    loss_mask = self.loss_mask(mask_pred, mask_targets,labels)
        loss['loss_mask'] = loss_mask
        return loss

    @force_fp32(apply_to=('mask_pred', ))
    def loss_crossfield_grad_align(self, mask_pred, pre_cossfields,pos_rois):
        loss = dict()
        if mask_pred.size(0) == 0 or (self.loss_roof_crossfield is None and self.loss_roof_facade_horiontal is None):
            loss_roof_crossfield = torch.tensor(0.)
            loss_roof_facade_horiontal=torch.tensor(0.)
        else:
            pre_cossfields_roi=self.crossfield_roi_align(pre_cossfields,pos_rois)
            mask_grad=self.grad_calculator(mask_pred)
            # loss_roof_facade_horiontal=self.loss_roof_facade_horiontal(mask_grad) if self.loss_roof_facade_horiontal is not None else torch.tensor(0.)
            loss_roof_crossfield=self.loss_roof_crossfield(mask_grad, pre_cossfields_roi) if self.loss_roof_crossfield is not None else torch.tensor(0.)
        loss['loss_roof_crossfield'] = loss_roof_crossfield
        # loss['loss_roof_facade_horiontal'] = loss_roof_facade_horiontal
        return loss

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg,instance_indicator):
        # FCNMaskCrossfieldConstraintHead重写了get_targets方法，新的get_targets方法可以接收一个新的instance_indicator参数，并传
        # 递给mask_target.mask_target方法，从而实现将从gt_mask中区分处各个instance对应的部分mask。从而避免之前的错误：直接用bbox从无区
        # 分的mask中截取，因而可能导致由于instance之间犬牙差互的位置关系而导致在一个bbox中截取到不属于这个instance的mask。
        # instance_indicator的具体实现方法为：instance_indicator是一个tensor[b, 256, 256]，类型为uint8，其中的每一个像素代表此像素
        # 属于哪一个instance，而背景处的instance_indicator为255。在提取mask时，corssfield对应的mask是整体的多类的tensor
        # mask，即每张图对应一个tensor[256，256，c]的mask，若具体进行target提取时（mask_target_single函数）输入了
        # instance_indicator参数，那么则会在怎加一步instance筛选操作，只有pos_assigned_gt_inds_list中的id与instance_indicator上的
        # id相匹配的部分mask才能被提取出来，从而实现单个instance的mask提取。
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg,instance_indicator)
        return mask_targets
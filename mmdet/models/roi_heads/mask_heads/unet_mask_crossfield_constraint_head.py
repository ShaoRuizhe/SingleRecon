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
from mmdet.core import force_fp32, mask_target, auto_fp16
from . import FCNMaskCrossfieldConstraintHead
from .fcn_mask_crossfield_constraint_head import ComputeSegGrads
from ...builder import HEADS, build_loss, build_roi_extractor
from .fcn_mask_head import FCNMaskHead
from mmcv import ops
from mmdet.models.dense_heads.unet_head import DoubleConv, OutConv

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit


class Up(nn.Module):
    """Upscaling then double conv
    input:x1:[b,in_channel,w,h] x2:[b,in_channel/2,2w,2h] output:[b,out_channel,2w,2h]
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


@HEADS.register_module()
class UNetMaskCrossfieldConstraintHead(FCNMaskHead):

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
                 loss_roof_crossfield=None,
                 loss_roof_facade_horiontal=None,
                 ):
        super(UNetMaskCrossfieldConstraintHead, self).__init__(num_convs,  # todo:调整参数输入
                                                               roi_feat_size,
                                                               32,  # in_channels
                                                               conv_kernel_size,
                                                               32,  # conv_out_channels
                                                               num_classes,
                                                               class_agnostic,
                                                               upsample_cfg,
                                                               conv_cfg,
                                                               norm_cfg,
                                                               loss_mask)
        self.up1 = (Up(256, 128, bilinear=False))  # out[-1]=2*mid
        self.up2 = (Up(128, 64, bilinear=False))   # todo:接入config调整此网络参数
        self.up3 = (Up(64, 32, bilinear=False))
        self.up4 = (Up(32, 16, bilinear=False))
        self.compress_convs = nn.ModuleList()
        for channel in [32, 64, 128, 256]:
            self.compress_convs.append(
                nn.Sequential(
                    nn.Conv2d(256, channel, kernel_size=3, padding=1, bias=False),  # 新进来的要压缩维度
                    nn.BatchNorm2d(channel),
                    nn.ReLU(inplace=True), ))
        self.outc = (OutConv(32, 32))

        self.loss_roof_crossfield = build_loss(loss_roof_crossfield) if loss_roof_crossfield is not None else None
        self.loss_roof_facade_horiontal = build_loss(
            loss_roof_facade_horiontal) if loss_roof_facade_horiontal is not None else None
        self.grad_calculator = ComputeSegGrads(
            device=torch.device('cuda'))  # 使用torch.cuda.set_device(int_X/'cuda:X')之后，torch.device('cuda')会自动获取到前面指定的卡
        self.crossfield_roi_aligns = self.build_roi_layers(crossfield_roi_align)

    def build_roi_layers(self, layer_cfg):
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
            list[n_level] of 多个尺寸crossfield_roi_align，尺寸对应如下：
                map.
                scale	feat_out
                3		224
                2		112
                1		56
                0		28
        """

        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        output_sizes = cfg.pop('output_sizes')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        crossfield_roi_aligns = []
        for output_size in output_sizes:
            crossfield_roi_aligns.append(layer_cls(output_size=output_size, **cfg))
        return crossfield_roi_aligns

    @auto_fp16()
    def forward(self, x):
        """

        Args:
            x: list[n_scale=4] of list[n_level=1~4] of tensor[n_obj,c=256,h_level,w_level] 其中scale分组和各个scale包含的level情况如下：
            levelx括号内为从resnet backbone获取的feature尺寸，下面的数字即为不同scale内不同level的feature的尺寸，strides那一行为特征尺寸与图像尺寸之笔，ROIAlign中会用到
                		level0(256)	level1(128)	level2(64)	level3(32)	feature_out
                strides	4	    	8	    	16	    	32
            scale
            3		    112		    56	    	28	    	14	    	224
            2		    56		    28	    	14		         		112
            1		    28	    	14		        	    			56
            0		    14							                	28

        Returns:
            mask_pred:list[n_scale=4] of tensor[n_obj,h_scale,w_scale] 尺寸情况如上表的feature out一栏
        """
        logits = [[]] * len(x)
        mask_pred = [torch.tensor([])] * len(x)
        for single_scale_x in x:
            for i, level_x in enumerate(single_scale_x):
                single_scale_x[i] = self.compress_convs[i](level_x)
        if len(x[3]) > 0:
            x2, x3, x4, x5 = x[3]
            x_l1 = self.up1(x5, x4)
            x_l1 = self.up2(x_l1, x3)
            x_l1 = self.up3(x_l1, x2)
            # x = self.up4(x, x1)# todo:直接加入图像作为最大层特征
            logits[3] = self.outc(x_l1)
        if len(x[2]) > 0:
            x2, x3, x4 = x[2]
            x_l2 = self.up2(x4, x3)
            x_l2 = self.up3(x_l2, x2)
            # x = self.up4(x, x1)
            logits[2] = self.outc(x_l2)
        if len(x[1]) > 0:
            x2, x3 = x[1]
            x_l3 = self.up3(x3, x2)
            # x = self.up4(x, x1)
            logits[1] = self.outc(x_l3)
        if len(x[0]) > 0:
            x_l4, = x[0]
            # x = self.up4(x, x1)
            logits[0] = self.outc(x_l4)
        for i, x in enumerate(logits):
            if len(x) > 0:
                for conv in self.convs:
                    x = conv(x)
                if self.upsample is not None:
                    x = self.upsample(x)
                    if self.upsample_method == 'deconv':
                        x = self.relu(x)
                mask_pred[i] = self.conv_logits(x)
        return mask_pred

    @force_fp32(apply_to=('mask_pred',))
    def loss(self, mask_pred, mask_targets, labels):
        # 分level分别计算ce mask loss 输入的结果为按照像素数量的平均
        loss = dict()
        loss_mask = 0
        for single_scale_mask_pred, single_scale_mask_targets in zip(mask_pred, mask_targets):
            if single_scale_mask_pred.size(0) == 0:
                loss_mask += single_scale_mask_pred.sum() * 0
            else:
                if self.class_agnostic:
                    loss_mask += self.loss_mask(single_scale_mask_pred, single_scale_mask_targets,
                                                torch.zeros_like(labels))
                else:
                    loss_mask += self.loss_mask(single_scale_mask_pred, single_scale_mask_targets,
                                                reduction_override='sum')
        loss['loss_mask'] = loss_mask / sum(level_mask.numel() for level_mask in mask_pred)
        return loss

    @force_fp32(apply_to=('mask_pred',))
    def loss_crossfield_grad_align(self, mask_pred, pre_cossfields, multi_level_pos_rois):
        """
        分level分别提取crossfield计算align loss 输入的结果为按照像素数量的平均
        Args:
            mask_pred:
            pre_cossfields:
            multi_level_pos_rois:

        Returns:

        """
        loss = dict()
        num_levels = len(multi_level_pos_rois)
        if sum([scale_mask_pred.size(0) for scale_mask_pred in mask_pred]) == 0 or (
                self.loss_roof_crossfield is None and self.loss_roof_facade_horiontal is None):
            loss_roof_crossfield = torch.tensor(0.)
            loss_roof_facade_horiontal = torch.tensor(0.)
        else:
            loss_roof_crossfield = 0
            for i, level_pos_rois in enumerate(multi_level_pos_rois):
                if len(mask_pred[i]) > 0:
                    pre_cossfields_roi = self.crossfield_roi_aligns[i](pre_cossfields, level_pos_rois)
                    mask_grad = self.grad_calculator(mask_pred[i])
                    # loss_roof_facade_horiontal = self.loss_roof_facade_horiontal(
                    #     mask_grad) if self.loss_roof_facade_horiontal is not None else torch.tensor(0.)
                    loss_roof_crossfield += self.loss_roof_crossfield(mask_grad, pre_cossfields_roi, reduction='sum') \
                        if self.loss_roof_crossfield is not None else torch.tensor(0.)
        loss['loss_roof_crossfield'] = loss_roof_crossfield / sum(
            [scale_mask[:, self.loss_roof_crossfield.pre_channel].numel() for scale_mask in mask_pred if
             len(scale_mask) > 0])
        # loss['loss_roof_facade_horiontal'] = loss_roof_facade_horiontal
        return loss

    def get_multi_scale_targets(self, sampling_results, level_ids, gt_masks, rcnn_train_cfg, instance_indicator):
        """
        对于sampling_results输入的多个尺寸的bbox，从gt_masks中提取其对应的mask标签。
        具体实现方式为根据输入的level_id，将sampling_results中的bbox进行level分组（list[n_level] of list[n_imgs] of tensor)，
        然后分level输入到mask_target函数中，mask_target函数会根据再分别对每个img的数据调用single_mask_target函数提取mask，并合并输出。
        最终的输出即为分level的标签。FCNMaskCrossfieldConstraintHead重写了get_targets方法实现的instance_indicator依然进行了保留。
        Args:
            sampling_results:
            level_ids:
            gt_masks:
            rcnn_train_cfg:
            instance_indicator:

        Returns:
            list[n_level] of tensor[n_level,c=3,h_level,h_level]  其中level（表中scale）和输出尺寸（表中（feat_out)的对应关系如下：
                scale	feat_out
                3		224
                2		112
                1		56
                0		28
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]

        level_mask_targets = []
        for level, single_level_ids in enumerate(level_ids):
            # 分层内的pos和id区分图像来源
            id_start = 0
            single_level_pos_proposals = []
            single_level_assigned_gt_inds = []
            for i, pos_proposals_single_img in enumerate(pos_proposals):
                single_img_level_ids = single_level_ids[single_level_ids >= id_start]
                single_img_level_ids = single_img_level_ids - id_start
                single_img_level_ids = single_img_level_ids[single_img_level_ids < len(pos_proposals_single_img)]
                single_level_pos_proposals.append(pos_proposals_single_img[single_img_level_ids])
                single_level_assigned_gt_inds.append(pos_assigned_gt_inds[i][single_img_level_ids])
                id_start += len(pos_proposals_single_img)
            cfg = rcnn_train_cfg.copy()
            cfg.mask_size = cfg.mask_size[level]
            level_mask_targets.append(mask_target(single_level_pos_proposals, single_level_assigned_gt_inds,
                                                  gt_masks, cfg, instance_indicator))
        return level_mask_targets

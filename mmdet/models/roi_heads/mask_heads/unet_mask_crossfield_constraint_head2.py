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
from . import FCNMaskCrossfieldConstraintHead, UNetMaskCrossfieldConstraintHead
from .fcn_mask_crossfield_constraint_head import ComputeSegGrads
from .unet_mask_crossfield_constraint_head import Up
from ...builder import HEADS, build_loss, build_roi_extractor
from .fcn_mask_head import FCNMaskHead
from mmcv import ops
from mmdet.models.dense_heads.unet_head import DoubleConv, OutConv

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit



@HEADS.register_module()
class UNetMaskCrossfieldConstraintHead2(UNetMaskCrossfieldConstraintHead):

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
        super(UNetMaskCrossfieldConstraintHead, self).__init__(# 调用 super's super FCNMaskHead 的 __init__
                num_convs=num_convs,
                 roi_feat_size=roi_feat_size,
                 in_channels=in_channels,
                 conv_kernel_size=conv_kernel_size,
                 conv_out_channels=conv_out_channels,
                 num_classes=num_classes,
                 class_agnostic=class_agnostic,
                 upsample_cfg=upsample_cfg,
                 conv_cfg=conv_cfg,
                 norm_cfg=norm_cfg,
                 loss_mask=loss_mask,)
        self.up1 = (Up(2048, 1024, bilinear=False))
        self.up2 = (Up(1024, 512, bilinear=False))   # todo:接入config调整此网络参数
        self.up3 = (Up(512, 256, bilinear=False))
        self.up4 = (Up(256, 256, bilinear=False))
        self.outc = (OutConv(256, 256))

        self.loss_roof_crossfield = build_loss(loss_roof_crossfield) if loss_roof_crossfield is not None else None
        self.loss_roof_facade_horiontal = build_loss(
            loss_roof_facade_horiontal) if loss_roof_facade_horiontal is not None else None
        self.grad_calculator = ComputeSegGrads(
            device=torch.device('cuda'))  # 使用torch.cuda.set_device(int_X/'cuda:X')之后，torch.device('cuda')会自动获取到前面指定的卡
        self.crossfield_roi_aligns = self.build_roi_layers(crossfield_roi_align)

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



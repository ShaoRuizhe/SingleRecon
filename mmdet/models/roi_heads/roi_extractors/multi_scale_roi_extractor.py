# encoding:utf-8
import torch
import torch.nn as nn
from mmcv import ops
from abc import ABCMeta, abstractmethod

from mmdet.core import force_fp32
from mmdet.models.builder import ROI_EXTRACTORS
from .single_level_roi_extractor import SingleRoIExtractor


@ROI_EXTRACTORS.register_module()
class MultiScaleRoIExtractor(nn.Module, metaclass=ABCMeta):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56):
        """

        Args:
            roi_layer:注意，这里layer_cfg中的output_size和featmap_strides的顺序：，output_size和featmap_strides都是从大尺寸到小尺寸，
            与表中level的存放顺序一致。但是构建时output_size的后n个对应featmap_strides的前n个。
            out_channels:
            featmap_strides:
            finest_scale:
        """
        super(MultiScaleRoIExtractor, self).__init__()
        self.roi_layers = self.build_pyramid_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.fp16_enabled = False
        self.finest_scale = finest_scale

    @property
    def num_inputs(self):
        """int: Number of input feature maps."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_pyramid_roi_layers(self,layer_cfg,featmap_strides):
        """

        Args:
            layer_cfg:
            featmap_strides:
        Returns:
            roi_layers:list[n_scale=4] of ModuleList[num_features_in_this_level]
            如下表所示，roi_layers是一个list，其第0，1，2，3项分别是有1，2，3，4个RoiAlign的modulelist:

                        level0(256)	level1(128)	level2(64)	level3(32)	feature_out
                strides	4	    	8	    	16	    	32
            scale
            3		    112		    56	    	28	    	14	    	224
            2		    56		    28	    	14		         		112
            1		    28	    	14		        	    			56
            0		    14							                	28
        """
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        sampling_ratio=cfg.sampling_ratio
        assert 'output_sizes' in cfg
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)

        roi_layers=[]
        levels=len(cfg.output_sizes)
        for i in range(levels):
            level_roi_layers=nn.ModuleList(
            [layer_cls(spatial_scale=1 / s,output_size=size,sampling_ratio=sampling_ratio) for s,size in zip(featmap_strides[:i+1],cfg.output_sizes[levels-i-1:])])
            roi_layers.append(level_roi_layers)
        return roi_layers

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function.
        计算rois的尺度，并根据尺度调用不同的roi_align组来提取特征。返回提取的特征和尺度划分后的rois和划分后与划分前的id对应关系
        Returns:
            pyramid_roi_feats: list[n_level] of list[num_feature_of_this_level] of tensor[n_roi_of_this_level,256,w,h of this feature_level]
            level_ids:list[n_level] of tensor[n_level_obj],dtype=int64 which roi is in this level
            level_rois:list[n_level] of roi_tensor[n_level_obj,5]

        """
        num_levels = len(feats)
        level_rois,level_ids = self.map_roi_levels(rois, num_levels)

        pyramid_roi_feats=[[]]*num_levels
        for i in range(num_levels):
            if len(level_rois[i])>0:
                pyramid_roi_feats[i]=[level_roi_layer(feats[level],level_rois[i]) for level,level_roi_layer in enumerate(self.roi_layers[i])]
        return pyramid_roi_feats,level_rois,level_ids

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            level_ids:list[n_level] of tensor[n_level_obj],dtype=int64 which roi is in this level
            level_rois:list[n_level] of roi_tensor[n_level_obj,5]

        Returns:
            XX
        """
        level_rois=[]
        level_ids=[]
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        for level in range(num_levels):
            single_level_ids_list=torch.argwhere(target_lvls==level).squeeze(1)
            level_rois.append(rois[single_level_ids_list])
            level_ids.append(single_level_ids_list)
        return level_rois,level_ids
# -*- encoding: utf-8 -*-
'''
@File    :   offset_roi_head.py
@Time    :   2021/01/17 21:10:35
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2021
@Desc    :   RoI head for offset model training
'''

import numpy as np
import torch
from abc import abstractmethod

from mmdet.core import bbox2roi, bbox2result, roi2bbox
from .loft_roi_head import LoftRoIHead
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead
from .test_mixins import OffsetTestMixin


@HEADS.register_module()
class UNetRoIHead(LoftRoIHead,StandardRoIHead, OffsetTestMixin):

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing.

        Returns:
            mask_results:dict{'mask_pred':list[n_level=4] of  tensor[n_level_obj,h_level,w_level,'mask_feats'：list[n_level=4] of tesnor]
            level_rois:list[n_level] of roi_tensor[n_level_obj,5]
            level_ids:list[n_level] of tensor[n_level_obj],dtype=int64
            其中mask尺寸情况如下：
                scale	feat_out
                3		224
                2		112
                1		56
                0		28

        """
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats,level_rois,level_ids = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]
        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results, level_rois,level_ids



    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """
        分level分别处理mask（核心为past_mask),然后合并到一起。
        输出：list[1] of array[n_objs,c=3,img_h=1024,img_w=1024] 其中外层的list是为了兼容以往模型的输出
        Simple test for mask head without augmentation."""
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device).to(torch.float32)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_results,level_rois,level_ids = self._mask_forward(x, mask_rois)
            # time0=time.time()
            # segm_results=np.zeros((len(mask_rois),ori_shape[2],*ori_shape[:2])) # 不要这样先构建，在查询，这样mask太大了会导致查询耗时长。改进为用list实现的以下方法：
            segm_results = [[]] * len(mask_rois)
            for level, mask_result in enumerate(mask_results['mask_pred']):
                if len(mask_result) > 0:
                    level_segm_result = self.mask_head.get_seg_masks(
                        mask_result, level_rois[level][:, 1:], det_labels, self.test_cfg,
                        ori_shape, scale_factor, rescale)
                    single_level_ids = level_ids[level].cpu().numpy()
                    for i, result in enumerate(level_segm_result[0]):
                        segm_results[single_level_ids[i]] = result
            segm_results = np.conjugate(np.array(segm_results))
            # print('get_seg_masks time:',time.time()-time0)
        return [segm_results.astype(bool)]

    def _mask_forward_train(self, x, sampling_results,bbox_feats, gt_masks,
                            img_metas,pre_cossfield=None,**kwargs):
        """Run forward function and calculate loss for mask head in
        training.
        与以往的roi_head不同之处在于：除了mask_results，还需要从_mask_forward获取level_rois,level_ids。其中level_ids用于将不同level的
        bbox与gt_masks对应，level_rois用于将roi按照level存放，之后提取分level的crossfield"""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results,level_rois,level_ids = self._mask_forward(x, pos_rois)
        else:
            raise NotImplementedError

        mask_targets = self.mask_head.get_multi_scale_targets(sampling_results,level_ids, gt_masks,
                                                  self.train_cfg,**kwargs)# 需要使用这个cfg中的cfg.mask_size
        # 为了兼容以前的无instance_indicator的普通mask，这里及以前都用**kwargs来代表可选输入instance_indicator
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)
        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        if pre_cossfield is not None:
            # crossfield_targets = self.mask_head.get_targets(sampling_results, pre_cossfield,
            #                                       self.train_cfg)
            loss_crossfield = self.mask_head.loss_crossfield_grad_align(mask_results['mask_pred'], pre_cossfield,level_rois)
            mask_results['loss_mask'].update(loss_crossfield)
        return mask_results
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
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead
from .test_mixins import OffsetTestMixin


@HEADS.register_module()
class LoftRoIHead(StandardRoIHead, OffsetTestMixin):
    def __init__(self,
                offset_roi_extractor=None,
                offset_head=None,
                **kwargs):
        assert offset_head is not None
        super(LoftRoIHead, self).__init__(**kwargs)

        if offset_head is not None:
            self.init_offset_head(offset_roi_extractor, offset_head)

        self.with_vis_feat = False

    def init_offset_head(self, offset_roi_extractor, offset_head):
        self.offset_roi_extractor = build_roi_extractor(offset_roi_extractor)
        self.offset_head = build_head(offset_head)

    def init_weights(self, pretrained):
        super(LoftRoIHead, self).init_weights(pretrained)
        self.offset_head.init_weights()

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      pre_cossfield=None,
                      gt_offsets=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposals (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        bbox_results = {
            'bbox_feats': None}  # config中有mask_roi_extractor，那么 self.share_roi_extractor就为false，_mask_forward_train和_offset_forward_train中会自己提取feat，不会用到输入的bbox_feat todo：将bbox这种之后可能要用到的弄到self里面，不要用函数参数传输
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas,pre_cossfield=pre_cossfield,**kwargs)
            # TODO: Support empty tensor input. #2280
            if mask_results['loss_mask'] is not None:
                losses.update(mask_results['loss_mask'])

        if self.with_offset:
            # print("mask_results['mask_pred']: ", mask_results['mask_pred'].shape)
            # print("mask_results['mask_targets']: ", mask_results['mask_targets'].shape)
            # print("bbox_results['bbox_feats']: ", bbox_results['bbox_feats'].shape)
            offset_results = self._offset_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_offsets, img_metas)
            # TODO: Support empty tensor input. #2280
            if offset_results['loss_offset'] is not None:
                losses.update(offset_results['loss_offset'])

        return losses

    def _offset_forward_train(self, 
                              x, 
                              sampling_results, 
                              bbox_feats, 
                              gt_offsets,
                              img_metas):
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        # if pos_rois.shape[0] == 0:
        #     return dict(loss_offset=None)
        offset_results = self._offset_forward(x, pos_rois)

        offset_targets = self.offset_head.get_targets(sampling_results, gt_offsets,
                                                  self.train_cfg)

        loss_offset = self.offset_head.loss(offset_results['offset_pred'], offset_targets)

        offset_results.update(loss_offset=loss_offset, offset_targets=offset_targets)
        return offset_results

    def _offset_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            offset_feats = self.offset_roi_extractor(
                x[:self.offset_roi_extractor.num_inputs], rois)
        else:
            assert bbox_feats is not None
            offset_feats = bbox_feats[pos_inds]

        # self._show_offset_feat(rois, offset_feats)

        offset_pred = self.offset_head(offset_feats)
        offset_results = dict(offset_pred=offset_pred, offset_feats=offset_feats)
        return offset_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas,pre_cossfield=None,**kwargs):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg,**kwargs)# 需要使用这个cfg中的cfg.mask_size
        # 为了兼容以前的无instance_indicator的普通mask，这里及以前都用**kwargs来代表可选输入instance_indicator
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)
        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        if pre_cossfield is not None:
            # crossfield_targets = self.mask_head.get_targets(sampling_results, pre_cossfield,
            #                                       self.train_cfg)
            loss_crossfield = self.mask_head.loss_crossfield_grad_align(mask_results['mask_pred'], pre_cossfield,pos_rois)
            mask_results['loss_mask'].update(loss_crossfield)
        return mask_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x:
            proposal_list:
            img_metas:
            proposals:
            rescale:
        Returns:
            result:dict{根据模型是否具有以下几种head，输出可以包括：（如果没有head，result就不存在这一条数据）
                'bbox':bbox_pre:list[1](ndarray[n,5]) 第一个list的1代表目标类别，这里只有一个类别，即建筑物；n个建筑物框，每个框4个坐标( in [tl_x, tl_y, br_x, br_y] format )和一个score
                'segm':
                    segm对于两种网络分为两种情况：
                    1.旧网络的输出为cls形式的mask，segm的形式为：list[1] of list [n] of dict{count}。n个建筑物每个一个segm，segm的形式是与原图像相同大小的1024*1024的mask
                    2.新网络是多类分割，其segm形式为：list[3],但是只有第一个项具有数据，其余项为空。第一项为list[n_objs] of ndarray[3,1024,1024]。以此形式存储是由于offnadir方法是将一个类别的目标分割出多个mask，
                    并且用一个多channel的mask记录这个多重mask，因此segm只有一个类别，但是FCN内部构建的时候是根据mask的类别数来构建的，因此输出就变成了长度为3（mask类别数）的list，但是只有第一个list内存储了多重mask的array
                'offset':offset_pre：ndarray[n,2],每个建筑物一个offset，offset分为x方向和y方向两个值
                }
        """
        # assert self.with_bbox, 'Bbox head must be implemented.'
        bbox_results,segm_results,offset_results=None,None,None
        result={'bbox':None,'segm':None,'offset':None}
        # 将输出的bbox初始化为rpn的输出
        det_bboxes=proposal_list[0][:1024]
        det_labels=torch.zeros(det_bboxes.shape[0],dtype=int)
        bbox_results = bbox2result(det_bboxes, det_labels,1)
        result['bbox']=bbox_results
        # import matplotlib.pyplot as plt
        # plt.imshow(plt.imread(img_metas[0]['filename']))
        # for i in range(100):
        #     bbox = bbox_results[0][i, :4]
        #     ax = plt.gca()
        #     ax.add_patch(
        #         plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], color="blue", fill=False,
        #                       linewidth=1))
        #
        # plt.show()
        if 'bbox' in result:
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
            bbox_results = bbox2result(det_bboxes, det_labels,
                                       self.bbox_head.num_classes)
            result['bbox'] = bbox_results
            # import matplotlib.pyplot as plt
            # plt.imshow(plt.imread(img_metas[0]['filename']))
            # for i in range(100):
            #     bbox = bbox_results[0][i, :4]
            #     ax = plt.gca()
            #     ax.add_patch(
            #         plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], color="blue", fill=False,
            #                       linewidth=1))
            #
            # plt.show()
        if 'segm' in result:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            result['segm'] = segm_results
            # import matplotlib.pyplot as plt
            # image = plt.imread(img_metas[0]['filename'])
            # for i in range(10):
            #     image[:, :, 0][segm_results[0][i]] += image[:, :, 0][segm_results[0][i]] * 0.8
            # plt.imshow(image)
            # plt.show()

        if 'offset' in result:
            if self.with_vis_feat:# todo
                offset_results = self.simple_test_offset_rotate_feature(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return bbox_results, segm_results, offset_results, self.vis_featuremap
            else:
                offset_results = self.simple_test_offset(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                result['offset'] = offset_results
            # import matplotlib.pyplot as plt
            # plt.imshow(plt.imread(img_metas[0]['filename']))
            # for i in range(25):
            #     bbox = bbox_results[0][i, :4]
            #     ax = plt.gca()
            #     ax.add_patch(
            #         plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], color="blue",
            #                       fill=False,
            #                       linewidth=1))
            #     ax.arrow((bbox[0] + bbox[2]) / 2, bbox[3], offset_results[i, 0], offset_results[i, 1])
            # plt.show()
        return result
        # return bbox_results,segm_results,offset_results


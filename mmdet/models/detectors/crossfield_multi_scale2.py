# encoding:utf-8
import mmcv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict

from frame_field_learning.losses import ComputeSegGrads
from .crossfield_multi_scale import CrossfieldMultiScale
from ..builder import DETECTORS,build_head,build_loss
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class CrossfieldMultiScale2(CrossfieldMultiScale):

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_crossfields=None,
                      gt_segs=None,
                      proposals=None,
                      **kwargs):
        """

        Args:
            img:
            img_metas:
            gt_bboxes:
            gt_labels:
            gt_bboxes_ignore:
            gt_masks:tensor[b,3,1024,1024] 3channels:interior(in range 0,1,2),edge(in range 0,1,2,3),vertex(in range 0,1) 0-background
            gt_crossfield:tensor[b,256,256]
            proposals:
            **kwargs:

        Returns:

        """
        bb_x = self.backbone(img)
        losses = dict()
        pre_cossfield=None
        if self.crossfield_head is not None:
            gt_field = torch.cat([torch.cos(gt_crossfields[:,None,...]),
                                  torch.sin(gt_crossfields[:,None,...])], dim=1)
            crossfield_losses,pre_cossfield=self.crossfield_head.forward_train(bb_x,gt_field,gt_segs)
            losses.update(crossfield_losses)
        if self.rpn_head is not None:
            if self.with_neck:
                x = self.neck(bb_x)
            # RPN forward and loss
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
            if self.roi_head is not None:
                # roi_losses = self.roi_head.forward_train(x,bb_x, img_metas, proposal_list,
                #                                          gt_bboxes, gt_labels,
                #                                          gt_bboxes_ignore, gt_masks=gt_segs,pre_cossfield=pre_cossfield,
                #                                          **kwargs)
                roi_losses = self.roi_head.forward_train(x,bb_x, img_metas, proposal_list,
                                                         gt_bboxes, gt_labels,
                                                         gt_bboxes_ignore, gt_masks=gt_masks,pre_cossfield=pre_cossfield,
                                                         **kwargs)# only roof
                losses.update(roi_losses)
        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """

        Args:
            img:
            img_metas:
            proposals:
            rescale:

        Returns: list[batch] of dict{'bbox','segm','offset',('crossfield'),('polygons')}
            'segm':list[3] 其中3是三那个类别，只有第一个类别存储了数据。'segm'[0]:dict('len':len,'bits':ndarray[len*3*1024*1024/8])，这是一个经过packbits的array，pack之后有效地节约了内存。
            roi_head输出的原尺寸为len，3，1024，1024，在本函数中进行了pack。可以通过np.unpackbits(np.packbits(segm)).reshape((len，3,1024,1024))恢复到原本的array。
            其中，len为目标数量，3为3个类别（background,facade,roof),1024是图像尺寸

            'polygons:list[2]:[polygons,probs]
                其中polygons：list[n_objs] of
                    dict{'tol_0.125':list[n_polygons] of shapely.geometry.Polygon,
                        'tol_1':list[n_polygons] of shapely.geometry.Polygon}
                    是各个segm的roof经过通过polygonize_post_process设定的polygonize方法矢量化得到的多边形,再经过不同tolerance(也通过polygonize_post_process设定) simplify之后得到的一个或多个多边形

        """
        bb_x = self.backbone(img)
        result = []
        pre_cossfield=None
        if self.crossfield_head is not None:
            pre_cossfield=self.crossfield_head(bb_x)
            pre_cossfield = F.interpolate(pre_cossfield, size=(1024, 1024))
        if self.rpn_head is not None:
            if self.with_neck:
                x = self.neck(bb_x)

            if proposals is None:
                proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals

            for i in range(x[0].shape[0]):# roi head一个一个图像处理，这是由于使用了RoIAlign，只能使用单个图像
                single_img_feat=tuple([x_level[i:i+1] for x_level in x])
                result_dict=self.roi_head.simple_test(
                    single_img_feat ,bb_x,proposal_list[i:i+1], img_metas[i:i+1], rescale=rescale)
                if self.polygonize_method is not None:
                    result_dict['polygons']=self.polygonize_method(np.array(result_dict['segm'][0])[:,2:])
                # result_dict['segm'][0]={'len':len(result_dict['segm'][0]),'bits':np.packbits(np.array(result_dict['segm'][0]))}
                if pre_cossfield is not  None:
                    result_dict['crossfield']=pre_cossfield[i].cpu().numpy()
                result.append(result_dict)


        else:
            for i in range(bb_x[0].shape[0]):
                result_dict={'crossfield':pre_cossfield[i].cpu().numpy()}
                result.append(result_dict)
        return result


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
from ..builder import DETECTORS,build_head,build_loss
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class CrossfieldMultiScale(TwoStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 crossfield_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_weights=None,
                 polygonize_post_process=None):
        """

        Args:
            backbone:
            neck:
            rpn_head:
            roi_head:
            crossfield_head:
            train_cfg:
            test_cfg:
            pretrained:
            loss_weights:
            polygonize_post_process: dict,测试时进行矢量化后处理的设置
        """
        super(CrossfieldMultiScale, self).__init__(
            backbone=backbone,
            neck=neck,# Unet of FPN
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.anchor_bbox_vis = [[287, 433, 441, 541]]
        self.with_vis_feat = True
        self.crossfield_head=build_head(crossfield_head) if crossfield_head is not None else None
        if rpn_head is None:
            self.rpn_head=None
        if loss_weights is None:
            self.loss_weights={}
        else:
            self.loss_weights=loss_weights
        self.polygonize_method=None
        if polygonize_post_process is not None:
            if polygonize_post_process.pop('method')=='simple':
                from frame_field_learning.polygonize_simple import polygonize
                self.polygonize_method=lambda x:polygonize(x,config=polygonize_post_process)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = 0
        for _key, _value in log_vars.items():
            if 'loss' in _key:
                if _key in self.loss_weights:
                    loss += _value * self.loss_weights[_key]
                else:
                    loss += _value

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

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
            gt_crossfield:tensor[b,1024,1024]
            proposals:
            **kwargs:

        Returns:

        """
        x = self.backbone(img)
        losses = dict()
        pre_cossfield=None
        if self.crossfield_head is not None:
            gt_field = torch.cat([torch.cos(gt_crossfields[:,None,...]),
                                  torch.sin(gt_crossfields[:,None,...])], dim=1)
            crossfield_losses,pre_cossfield=self.crossfield_head.forward_train(x,gt_field,gt_segs)
            losses.update(crossfield_losses)
        if self.rpn_head is not None:
            if self.with_neck:
                x = self.neck(x)
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
                # roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                #                                          gt_bboxes, gt_labels,
                #                                          gt_bboxes_ignore, gt_masks=gt_segs,pre_cossfield=pre_cossfield,
                #                                          **kwargs)
                roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                         gt_bboxes, gt_labels,
                                                         gt_bboxes_ignore, gt_masks=gt_masks,pre_cossfield=pre_cossfield,
                                                         **kwargs)
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
        x = self.backbone(img)
        result = []
        pre_cossfield=None
        if self.crossfield_head is not None:
            pre_cossfield=self.crossfield_head(x)
            pre_cossfield = F.interpolate(pre_cossfield, size=(1024, 1024))
        if self.rpn_head is not None:
            if self.with_neck:
                x = self.neck(x)

            if proposals is None:
                proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals

            for i in range(x[0].shape[0]):# roi head一个一个图像处理，这是由于使用了RoIAlign，只能使用单个图像
                single_img_feat=tuple([x_level[i:i+1] for x_level in x])
                result_dict=self.roi_head.simple_test(
                    single_img_feat, proposal_list[i:i+1], img_metas[i:i+1], rescale=rescale)
                if self.polygonize_method is not None:
                    result_dict['polygons']=self.polygonize_method(np.array(result_dict['segm'][0])[:,2:])
                # result_dict['segm'][0]={'len':len(result_dict['segm'][0]),'bits':np.packbits(np.array(result_dict['segm'][0]))}
                if pre_cossfield is not  None:
                    result_dict['crossfield']=pre_cossfield[i].cpu().numpy()
                result.append(result_dict)


        else:
            for i in range(x[0].shape[0]):
                result_dict={'crossfield':pre_cossfield[i].cpu().numpy()}
                result.append(result_dict)
        return result

    def draw_crossfield(self,crossfield,arrow_interval = 20):
        """

        Args:
            crossfield: tensor or np.ndarray
            arrow_interval: 1024-20 512-10
        """
        from frame_field_learning.frame_field_utils import c0c2_to_uv,c4c8_to_uv
        if isinstance(crossfield,np.ndarray):
            crossfield=torch.tensor(crossfield)
        if self.crossfield_head.loss_crossfield_align.level_2_align:
            uv = c4c8_to_uv(crossfield[None, ...])
        else:
            uv = c0c2_to_uv(crossfield[None,...])
        img_shape = uv.shape[-2:]
        arrow_len = 7
        # ref:https://juejin.cn/post/7034117434659471397
        x, y = np.meshgrid(np.arange(0, img_shape[0], arrow_interval), np.arange(0, img_shape[1], arrow_interval))
        uv_down = uv.cpu().detach().numpy()[:, :, :, ::arrow_interval, ::arrow_interval]
        u, v = (
        arrow_len * uv_down[0, 1, 0], arrow_len * uv_down[0, 1, 1])  # 由于绘制图像时还是图像上方y大，而quiver场显示是y的下方大，因此要对y方向的offset反向
        plt.quiver(x, y, u, v, color="pink", pivot="tail", units="inches")
        plt.scatter(x, y, color="b", s=0.05)
        # plt.imshow(img[425:625, 325:525])
        # from frame_field_learning.frame_field_utils import c0c2_to_uv, c4c8_to_uv
        # arrow_interval = 15
        # headwidth = 1
        # headlength = 1
        # width = 0.04
        # color = 'blue'
        # if isinstance(crossfield_result, np.ndarray):
        #     crossfield = torch.tensor(crossfield_result)[:, 425:625, 325:525]
        # if self.crossfield_head.loss_crossfield_align.level_2_align:
        #     uv = c4c8_to_uv(crossfield[None, ...])
        # else:
        #     uv = c0c2_to_uv(crossfield[None, ...])
        # img_shape = uv.shape[-2:]
        # arrow_len = 0.1
        # x, y = np.meshgrid(np.arange(0, img_shape[0], arrow_interval), np.arange(0, img_shape[1], arrow_interval))
        # uv_down = uv.cpu().detach().numpy()[:, :, :, ::arrow_interval, ::arrow_interval]
        # u, v = (
        #     arrow_len * uv_down[0, 1, 0],
        #     arrow_len * uv_down[0, 1, 1])  # 由于绘制图像时还是图像上方y大，而quiver场显示是y的下方大，因此要对y方向的offset反向
        # plt.quiver(x, y, u, v, color=color, pivot="tail", units="inches", width=width, headwidth=headwidth,
        #            headlength=headlength)
        # plt.quiver(x, y, -u, -v, color=color, pivot="tail", units="inches", width=width, headwidth=headwidth,
        #            headlength=headlength)
        # u, v = (
        #     arrow_len * uv_down[0, 0, 0],
        #     arrow_len * uv_down[0, 0, 1])  # 由于绘制图像时还是图像上方y大，而quiver场显示是y的下方大，因此要对y方向的offset反向
        # plt.quiver(x, y, u, v, color=color, pivot="tail", units="inches", width=width, headwidth=headwidth,
        #            headlength=headlength)
        # plt.quiver(x, y, -u, -v, color=color, pivot="tail", units="inches", width=width, headwidth=headwidth,
        #            headlength=headlength)
        # plt.scatter(x, y, color="b", s=0.05)
        # plt.show()

    def show_result(self,
                    img,
                    result,
                    score_thr=0.4,
                    num_obj=200,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """
        将目标置信度得分高于score_thr的显示出来
        Args:
            img:文件名或0~255 HWC BGR类型的图片
            result:上面的simple_test返回值result为一个batch的预测结果，而本函数的输入则为一个数据的预测结果，即result的一个项，result[i]。
            是一个dict，dict的内容与上面的simple_test一致：
                dict{'bbox','segm','offset',('crossfield')}
                    'segm':list[3] 其中3是三那个类别，只有第一个类别存储了数据。'segm'[0]:dict('len':len,'bits':ndarray[len*3*1024*1024/8])，这是一个经过packbits的array，pack之后有效地节约了内存。
                    原尺寸为len，3，1024，1024，可以通过np.unpackbits(np.packbits(segm)).reshape((len，3,1024,1024))恢复到原本的array。
                    其中，len为目标数量，3为3个类别（background,facade,roof),1024是图像尺寸
                    score_thr:bbox中的score（每个bbox中的最后一维）的thre，大于此score的才会被绘出
            num_obj:显示目标数量
            bbox_color:此参数及以下参数均未生效
            text_color:
            thickness:
            font_scale:
            win_name:
            show:
            wait_time:
            out_file:
        """
        # img = mmcv.imread(img)[:,:,::-1]# bgr->rgb
        img = img.copy()
        if isinstance(result, tuple):
            if self.with_vis_feat:
                # bbox_result, segm_result, offset, offset_features = result todo：offset_features是啥？
                bbox_result, segm_result, offset_result = result
            else:
                bbox_result, segm_result, offset_result = result
            inds = np.array(range(bbox_result[0].shape[0]))
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        elif isinstance(result,dict):
            bbox_result=result.get('bbox',None)
            segm_result=result.get('segm',None)
            offset_result=result.get('offset',None)
            crossfield_result=result.get('crossfield',None)
            inds = np.array(range(len(list(result.values())[0][0])))
        else:
            bbox_result, segm_result = result, None
            inds=np.array(range(bbox_result[0].shape[0]))
        if isinstance(offset_result, tuple):
            offset_result = offset_result[0]
        else:
            offset_result = offset_result

        # rotate offset
        # offsets = self.offset_rotate(offsets, 0)
        if bbox_result is not None:
            bbox_result = np.vstack(bbox_result)
            if bbox_result.shape[1]==5:
                scores = bbox_result[:, -1]
                bbox_result = bbox_result [:, 0:-1]
            else:
                scores=np.ones(len(bbox_result))
            w, h = bbox_result[:, 2] - bbox_result[:, 0], bbox_result[:, 3] - bbox_result[:, 1]
            area = w * h
            # valid_inds = np.argsort(area, axis=0)[::-1].squeeze()
            inds = np.where(np.bitwise_and(scores > score_thr ,np.sqrt(area) > 20))[0]
        # if segm_result is not None:  # non empty
        #     segm_result = mmcv.concat_list(segm_result)


        # 以下valid中记录result中满足面积阈值（area>50),并且score满足阈值（score0.4)的
        valid_segm_results = []
        valid_offset_results = []
        valid_bbox_results = []
        offset_feats = []
        img_temp = img.copy()
        if segm_result is not None:
            if isinstance(segm_result[0],dict):  # bite pack
                segm_result = np.unpackbits(segm_result[0]['bits']).reshape(segm_result[0]['shape']).astype(bool)
            else:
                segm_result = segm_result[0]  # origin
            for i in inds[:num_obj]:
            # for i in inds[6:7]:
                mask = segm_result[i]>0.5
                if len(mask.shape)==3:
                    # 对于offnadir三类：background，facade，roof
                    img_temp[:, :, 0][mask[1]] = (
                                img[:, :, 0][mask[1]] * 0.2 + 255 * np.ones((1024, 1024))[mask[1]] * 0.8).astype(
                        np.uint8)
                    img_temp[:, :, 1][mask[2]] = (
                            img[:, :, 1][mask[2]] * 0.2 + 255 * np.ones((1024, 1024))[mask[2]] * 0.8).astype(
                        np.uint8)
                elif len(mask.shape)==2:
                    img_temp[:, :, 0][mask] = (img[:, :, 0][mask] * 0.5 + 255 * np.ones((1024, 1024))[mask] * 0.5).astype(
                        np.uint8)
        plt.figure(figsize=(10,8),dpi=150)
        plt.imshow(img_temp)
        ax = plt.gca()
        for i in inds[:num_obj]:
            if offset_result is not None:
                offset = offset_result[i]
                if segm_result is not None:# todo:修改offset显示的位置设定
                    if len(segm_result[i].shape)==3:
                        segm_mask=segm_result[i][2]
                    else:
                        segm_mask=segm_result[i]
                    gray = np.array(segm_mask * 255, dtype=np.uint8)

                    # # if show_polygon:
                    polygonize_post_process = {
                        'method': 'simple',
                        "data_level": 0.5,
                        "tolerance": [0.125, 2],
                        "seg_threshold": 0.5,
                        "min_area": 10

                    }
                    from frame_field_learning.polygonize_simple import polygonize

                    polygon_roof = polygonize(segm_mask[None, None], polygonize_post_process)
                    if len(polygon_roof[0][0]['tol_2'])>0:
                        polygon_roof=polygon_roof[0][0]['tol_2'][np.argmax([poly.area for poly in polygon_roof[0][0]['tol_2']])]
                        plt.plot(*polygon_roof.exterior.xy, marker='o', color='red',markersize=2)

                        coors = polygon_roof.exterior.xy
                        right_point_id = np.argmin(coors[0])
                        end_point = [coors[0][right_point_id], coors[1][right_point_id]]
                        offset_point = end_point - offset
                    else:
                        offset_point = [0, 0]

                else:
                    offset_point=[0,0]
                offset_point=np.clip(offset_point,0,1024)
                ax.arrow(offset_point[0], offset_point[1], offset[0], offset[1], width=3, color='red',
                         length_includes_head=True)
            if bbox_result is not None:
                bbox=bbox_result[i]
                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], color="blue",
                                  fill=False,
                                  linewidth=1))
        if crossfield_result is not None:
            self.draw_crossfield(crossfield_result)
                # self.draw_crossfield(crossfield_result,arrow_interval=100)
            # if self.with_vis_feat:
            #     offset_feat = offset_features[i]
            # else:
            #     offset_feat = []

            # valid_segm_results.append(mask)
            # valid_offset_results.append(offset_result)
            # valid_bbox_results.append(bbox)
            # offset_feats.append(offset_feat)
        if out_file is not None:
            plt.savefig(out_file)
        if show:
            plt.show()
        plt.clf()
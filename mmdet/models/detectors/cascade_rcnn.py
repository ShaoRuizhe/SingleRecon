import mmcv
import numpy as np
import cv2
import matplotlib.pyplot as plt

from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class CascadeRCNN(TwoStageDetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CascadeRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    # def show_result(self, data, result, **kwargs):
    #     """Show prediction results of the detector."""
    #     if self.with_mask:
    #         ms_bbox_result, ms_segm_result = result
    #         if isinstance(ms_bbox_result, dict):
    #             result = (ms_bbox_result['ensemble'],
    #                       ms_segm_result['ensemble'])
    #     else:
    #         if isinstance(result, dict):
    #             result = result['ensemble']
    #     return super(CascadeRCNN, self).show_result(data, result, **kwargs)

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

        Args:
            img:文件名或0~255 HWC BGR类型的图片
            result: 父类TwoStageDetector.simple_test调用了LoftRoiHead.simple_test所返回的result，一个dict,详见LoftRoiHead.simple_test注释
            score_thr:bbox中的score（每个bbox中的最后一维）的thre，大于此score的才会被绘出
            num_obj:显示目标数量
            bbox_color:此参数及以下参数均未生效
            text_color:
            thickness:
            font_scale:
            win_name:
            show:
.            wait_time:
            out_file:
        """
        img = mmcv.imread(img)[:,:,::-1]# bgr->rgb
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

        if segm_result is not None:  # non empty
            segm_result = mmcv.concat_list(segm_result)
            # segm_result = mmcv.concat_list([segm_result])#todo:?`


        # 以下valid中记录result中满足面积阈值（area>50),并且score满足阈值（score0.4)的
        valid_segm_results = []
        valid_offset_results = []
        valid_bbox_results = []
        offset_feats = []
        img_temp = img.copy()
        for i in inds[:num_obj]:
        # for i in inds[6:7]:
            if segm_result is not None:
                mask = segm_result[i]>0.5
                img_temp[:, :, 0][mask] = (img[:, :, 0][mask] * 0.2 + 255 * np.ones((1024, 1024))[mask] * 0.8).astype(
                    np.uint8)
        plt.figure(figsize=(10,8),dpi=150)
        plt.imshow(img_temp)
        ax = plt.gca()
        for i in inds[:num_obj]:
            if offset_result is not None:
                offset = offset_result[i]
                if segm_result is not None:# todo:修改offset显示的位置设定
                    gray = np.array(mask * 255, dtype=np.uint8)

                    # # if show_polygon:
                    polygonize_post_process = {
                        'method': 'simple',
                        "data_level": 0.5,
                        "tolerance": [0.125, 2],
                        "seg_threshold": 0.5,
                        "min_area": 10

                    }
                    from frame_field_learning.polygonize_simple import polygonize

                    polygon_roof = polygonize(segm_result[i][None, None], polygonize_post_process)
                    if len(polygon_roof[0][0]['tol_2']) > 0:
                        polygon_roof = polygon_roof[0][0]['tol_2'][
                            np.argmax([poly.area for poly in polygon_roof[0][0]['tol_2']])]
                        plt.plot(*polygon_roof.exterior.xy, marker='o',color='red', markersize=2)

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
            # if self.with_vis_feat:
            #     offset_feat = offset_features[i]
            # else:
            #     offset_feat = []



            # valid_segm_results.append(mask)
            valid_offset_results.append(offset_result)
            valid_bbox_results.append(bbox)
            # offset_feats.append(offset_feat)
        if out_file is not None:
            plt.savefig(out_file)
        if show:
            plt.show()
# encoding:utf-8
import os
import os.path as osp
import math
import tempfile
import csv
import time
from multiprocessing import Pool

import numpy as np
from collections import defaultdict

import torch
from sklearn.metrics import mean_absolute_error
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm

from frame_field_learning.frame_field_utils import c4c8_to_uv
from lydorn_utils.polygon_utils import compute_polygon_contour_measures
from tools.polis import polis_metric, multi_polygon_polis_metric
from .coco import CocoDataset
from .builder import DATASETS
from .pipelines import Compose
from pycocotools import mask

from .utils import get_instance_seg_with_indicator, iou
from ..core import PolygonMasks
from tools.chamfer_distance import ChamferDistance


@DATASETS.register_module()
class BONAI(CocoDataset):
    CLASSES = ('building')

    def __init__(self,
                 ann_file,
                 pipeline,
                 anno_pipeline=None,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 edge_prefix=None,
                 side_face_prefix=None,
                 offset_field_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 gt_footprint_csv_file=None,
                 bbox_type='roof',
                 mask_type='roof',
                 offset_coordinate='rectangle',
                 resolution=0.6,
                 ignore_buildings=True):
        """

        Args:
            ann_file:
            pipeline:
            anno_pipeline:
            classes:
            data_root:
            img_prefix:
            seg_prefix:
            edge_prefix:
            side_face_prefix:
            offset_field_prefix:
            proposal_file:
            test_mode:
            filter_empty_gt:
            gt_footprint_csv_file:
            bbox_type:
            mask_type:
            offset_coordinate:
            resolution:
            ignore_buildings:
        """
        super(BONAI, self).__init__(ann_file=ann_file,
                                    pipeline=pipeline,
                                    classes=classes,
                                    data_root=data_root,
                                    img_prefix=img_prefix,
                                    seg_prefix=seg_prefix,
                                    proposal_file=proposal_file,
                                    test_mode=test_mode,
                                    filter_empty_gt=filter_empty_gt)
        self.ann_file = ann_file
        self.bbox_type = bbox_type
        self.mask_type = mask_type
        self.offset_coordinate = offset_coordinate
        self.resolution = resolution
        self.ignore_buildings = ignore_buildings
        self.gt_footprint_csv_file = gt_footprint_csv_file

        self.edge_prefix = edge_prefix
        self.side_face_prefix = side_face_prefix
        self.offset_field_prefix = offset_field_prefix

        if self.data_root is not None:
            if not (self.edge_prefix is None or osp.isabs(self.edge_prefix)):
                self.edge_prefix = osp.join(self.data_root, self.edge_prefix)

        if self.data_root is not None:
            if not (self.side_face_prefix is None or osp.isabs(self.side_face_prefix)):
                self.side_face_prefix = osp.join(self.data_root, self.side_face_prefix)

        if self.data_root is not None:
            if not (self.offset_field_prefix is None or osp.isabs(self.offset_field_prefix)):
                self.offset_field_prefix = osp.join(self.data_root, self.offset_field_prefix)
        if anno_pipeline is not None:
            self.anno_pipeline = Compose(anno_pipeline)
        else:
            self.anno_pipeline = None
        # print("This dataset has these keys: {}".format(list(self.get_properties(0))))

    def pre_pipeline(self, results):
        super(BONAI, self).pre_pipeline(results)
        results['edge_prefix'] = self.edge_prefix
        results['edge_fields'] = []

        results['side_face_prefix'] = self.side_face_prefix
        results['side_face_fields'] = []

        results['offset_field_prefix'] = self.offset_field_prefix
        results['offset_field_fields'] = []

    def get_properties(self, idx):
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)

        return ann_info[0].keys()

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            img_id = img_info['id']
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)
            if 'iscrowd' in ann_info[0]:
                all_iscrowd = all([_['iscrowd'] for _ in ann_info])
            else:
                all_iscrowd = False
            if self.filter_empty_gt and (self.img_ids[i] not in ids_with_ann
                                         or all_iscrowd):
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.
            根据bbox_type，将对应的bbox付给gt_bbox输出。如果bbox超出了图像范围，还会进行调整

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_roof_masks_ann = []
        gt_footprint_masks_ann = []
        gt_offsets = []
        gt_building_heights = []
        gt_angles = []
        gt_mean_angle = 0.0
        gt_roof_bboxes = []
        gt_footprint_bboxes = []
        gt_only_footprint_flag = 0

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            # bbox type may be roof, building and footprint, you need to set the value in config file
            if self.bbox_type == 'roof':
                x1, y1, w, h = ann['bbox']
            elif self.bbox_type == 'building':
                x1, y1, w, h = ann['building_bbox']
            elif self.bbox_type == 'footprint':
                x1, y1, w, h = ann['footprint_bbox']
            else:
                raise (TypeError(f"don't support bbox_type={self.bbox_type}"))

            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if 'area' in ann and (ann['area'] <= 0 or w < 1 or h < 1):
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False) and self.ignore_buildings:
                gt_bboxes_ignore.append(bbox)
            else:
                if 'roof_bbox' in ann:
                    x1, y1, w, h = ann['roof_bbox']
                    gt_roof_bboxes.append([x1, y1, x1 + w, y1 + h])
                if 'footprint_bbox' in ann:
                    x1, y1, w, h = ann['footprint_bbox']
                    gt_footprint_bboxes.append([x1, y1, x1 + w, y1 + h])
                if 'only_footprint' in ann:
                    if ann['only_footprint'] == 1:
                        gt_only_footprint_flag = 1
                    else:
                        gt_only_footprint_flag = 0

                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                # gt_only_footprint_flag=0: use roof as mask, gt_only_footprint_flag=0:use footprint as mask
                if gt_only_footprint_flag == 0:
                    if self.mask_type == 'roof':
                        if len(np.array(ann[
                                            'segmentation']).shape) == 1:  # 很坑，trainval中的是[[x,y,x,y]],test中的是[x,y,x,y]，自制的rotate也是一层 后续需要的是两层
                            gt_masks_ann.append([ann['segmentation']])
                        else:
                            gt_masks_ann.append(ann['segmentation'])
                    elif self.mask_type == 'footprint':
                        gt_masks_ann.append([ann['footprint_mask']])
                    else:
                        raise (TypeError(f"don't support mask_type={self.mask_type}"))
                else:
                    gt_masks_ann.append([ann['footprint_mask']])

                gt_roof_masks_ann.append([ann['segmentation']])
                if 'footprint_mask' in ann:
                    gt_footprint_masks_ann.append([ann['footprint_mask']])

                # rectangle coordinate -> offset = (x, y), polar coordinate -> offset = (length, theta)
                if 'offset' in ann:
                    if self.offset_coordinate == "rectangle":
                        gt_offsets.append(ann['offset'])
                    elif self.offset_coordinate == 'polar':
                        offset_x, offset_y = ann['offset']
                        length = math.sqrt(offset_x ** 2 + offset_y ** 2)
                        angle = math.atan2(offset_y, offset_x)
                        gt_offsets.append([length, angle])
                    else:
                        raise (RuntimeError(f'do not support this coordinate: {self.offset_coordinate}'))
                else:
                    gt_offsets.append([0, 0])

                if 'building_height' in ann:
                    gt_building_heights.append(ann['building_height'])
                else:
                    gt_building_heights.append(0.0)

                if 'offset' in ann and 'building_height' in ann:
                    offset_x, offset_y = ann['offset']
                    height = ann['building_height'] if 'building_height' in ann else 0
                    angle = math.atan2(math.sqrt(offset_x ** 2 + offset_y ** 2) * self.resolution, height)

                    gt_angles.append(angle)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_roof_bboxes = np.array(gt_roof_bboxes, dtype=np.float32)
            gt_footprint_bboxes = np.array(gt_footprint_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_offsets = np.array(gt_offsets, dtype=np.float32)
            gt_building_heights = np.array(gt_building_heights, dtype=np.float32)
            if len(gt_angles)>0:
                gt_mean_angle = float(np.array(gt_angles, dtype=np.float32).mean())
            else:
                gt_mean_angle=0.
            gt_only_footprint_flag = float(gt_only_footprint_flag)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_roof_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_footprint_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_offsets = np.zeros((0, 2), dtype=np.float32)
            gt_building_heights = np.zeros((0, 2), dtype=np.float32)
            gt_mean_angle = 0.0001
            gt_only_footprint_flag = 0

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')
        edge_map = img_info['filename'].replace('jpg', 'png')
        side_face_map = img_info['filename'].replace('jpg', 'png')
        offset_field = img_info['filename'].replace('png', 'npy')
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            roof_masks=gt_roof_masks_ann,
            footprint_masks=gt_footprint_masks_ann,
            seg_map=seg_map,
            offsets=gt_offsets,
            building_heights=gt_building_heights,
            angle=gt_mean_angle,
            pitch_ratio=ann_info[0]['pitch_ratio'] if len(ann_info)>0 else np.nan,
            edge_map=edge_map,
            side_face_map=side_face_map,
            roof_bboxes=gt_roof_bboxes,
            footprint_bboxes=gt_footprint_bboxes,
            offset_field=offset_field,
            only_footprint_flag=gt_only_footprint_flag)

        return ann

    def _segm2json(self, results):
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            if len(results[idx]) == 1:
                det = results[idx][0]
            elif len(results[idx]) == 2:
                det, seg = results[idx]
            elif len(results[idx]) == 3:
                det, seg, offset = results[idx]
            elif len(results[idx]) == 4:
                det, seg, offset, building_height = results[idx]
            else:
                raise (RuntimeError("do not support the length of results: ", len(results[idx])))

            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)
                if len(results[idx]) > 1:
                    # segm results
                    # some detectors use different scores for bbox and mask
                    if isinstance(seg, tuple):
                        segms = seg[0][label]
                        mask_score = seg[1][label]
                    else:
                        segms = seg[label]
                        mask_score = [bbox[4] for bbox in bboxes]
                    for i in range(bboxes.shape[0]):
                        data = dict()
                        data['image_id'] = img_id
                        data['bbox'] = self.xyxy2xywh(bboxes[i])
                        data['score'] = float(mask_score[i])
                        data['category_id'] = self.cat_ids[label]
                        if isinstance(segms[i]['counts'], bytes):
                            segms[i]['counts'] = segms[i]['counts'].decode()
                        data['segmentation'] = segms[i]
                        segm_json_results.append(data)

        return bbox_json_results, segm_json_results

    def write_results2csv(self, results, meta_info=None):
        print("meta_info: ", meta_info)
        segmentation_eval_results = results[0]
        with open(meta_info['summary_file'], 'w') as summary:
            csv_writer = csv.writer(summary, delimiter=',')
            csv_writer.writerow(['Meta Info'])
            csv_writer.writerow(['model', meta_info['model']])
            csv_writer.writerow(['anno_file', meta_info['anno_file']])
            csv_writer.writerow(['gt_roof_csv_file', meta_info['gt_roof_csv_file']])
            csv_writer.writerow(['gt_footprint_csv_file', meta_info['gt_footprint_csv_file']])
            csv_writer.writerow(['vis_dir', meta_info['vis_dir']])
            csv_writer.writerow([''])
            for mask_type in ['roof', 'footprint']:
                csv_writer.writerow([mask_type])
                csv_writer.writerow([segmentation_eval_results[mask_type]])
                csv_writer.writerow(['F1 Score', segmentation_eval_results[mask_type]['F1_score']])
                csv_writer.writerow(['Precision', segmentation_eval_results[mask_type]['Precision']])
                csv_writer.writerow(['Recall', segmentation_eval_results[mask_type]['Recall']])
                csv_writer.writerow(['True Positive', segmentation_eval_results[mask_type]['TP']])
                csv_writer.writerow(['False Positive', segmentation_eval_results[mask_type]['FP']])
                csv_writer.writerow(['False Negative', segmentation_eval_results[mask_type]['FN']])
                csv_writer.writerow([''])

            csv_writer.writerow([''])

    def prepare_test_gt(self, metrics):
        import mmdet.datasets.pipelines
        from mmdet.datasets.builder import PIPELINES
        from mmcv.utils import build_from_cfg
        load_args = {'type': 'LoadAnnotations', 'with_bbox': False, 'with_label': False, 'with_seg': False, }
        if 'angle' in metrics:
            load_args['with_angle'] = True
        if 'offset' in metrics:
            load_args['with_offset'] = True
        gt_annos = []  # todo:统一用anno_pipline,需要把bonai config文件都改一改
        if self.anno_pipeline is not None:
            for i in range(len(self)):
                results = dict(ann_info=self.get_ann_info(i), img_info=self.data_infos[i])
                self.pre_pipeline(results)
                results['bbox_fields'] = []
                results['mask_fields'] = []
                results['seg_fields'] = []
                results['offset_fields'] = []
                results['rbbox_fields'] = []
                gt_annos.append(self.anno_pipeline(results))
        else:
            anno_loader = build_from_cfg(load_args, PIPELINES)
            for i in range(len(self)):
                gt_annos.append(anno_loader(dict(ann_info=self.get_ann_info(i), offset_fields=[])))
        return gt_annos

    def evaluate(self,
                 results,
                 metric='bbox',
                 thr=0.5,
                 **kwargs):
        """
        用调用父类的cocoeval来评估bbox，segm（输出是cls形式，array形式的无法使用coco），然后用其他方法评估其他offset等其他指标
        Args:
            results: list[n_val_data](
                        dict{
                            bbox:list[1](ndarray[1024,5]),  ##:1个类别对应长度为1的list，1024对应取到的1024个框\n
                            segm:本函数可以处理以下两种segm形式：
                                对于原来的单类bonai数据，segm是以rle形式存储的，result['segm']的形式为：
                                    list[1] of list[1024] of
                                        dict{'size':list[2]=[1024,1024]
                                            'counts':btyes}
                                    )
                                对于crossfield多类别分割，roof_segm的类型为list[3]但是只有第一个项具有数据，其余项为空。
                                   'segm':list[3] 其中3是三那个类别，只有第一个类别存储了数据。'segm'[0]:dict('len':len,'bits':ndarray[len*3*1024*1024/8])，这是一个经过packbits的array，pack之后有效地节约了内存。
                                    原尺寸为len，3，1024，1024，可以通过np.unpackbits(np.packbits(segm)).reshape((len，3,1024,1024))恢复到原本的array。
                                    其中，len为目标数量，3为3个类别（background,facade,roof),1024是图像尺寸
                    )
            metric: list or str。要对那些指标进行计算 choise:'bbox', 'segm', 'proposal', 'proposal_fast'(这四个将会使用CoocDataset计算metric）,offset,angle
            **kwargs:

        Returns:

        """
        thred_results = []
        thr = thr
        for result in results:
            thre_ids = result['bbox'][0][:, 4] > thr
            thred_results.append({'bbox': [result['bbox'][0][thre_ids]],
                                'thre_ids': thre_ids,
                                'offset': result['offset'][thre_ids]})

        metrics = metric if isinstance(metric, list) else [metric]
        # coco bit类型存储的segm可以用coco来评估，ndarray类型的不能用coco，则计算一个自己编写的iou
        if 'segm' in results[0]:
            if isinstance(results[0]['segm'][0],list) :  # rle类型:list of dict [and]annotation is mask rather than polygon
                coco_metric = list({'bbox', 'proposal', 'proposal_fast', 'segm'} & set(metrics))
            else:  # array mask or bits(dict{shape,bit})
                coco_metric = list({'bbox', 'proposal', 'proposal_fast'} & set(metrics))
                # coco_metric = list({'bbox', 'proposal', 'proposal_fast', 'segm'} & set(metrics))
            coco_metric.sort()  # 需要按照一定顺序排序，因为pycocotools是按照一定顺序进来读取的
        else:
            coco_metric = []
        other_metrics = set(metrics) - set(coco_metric)  # 除了coco的评估指标，还需要评估其他指标，需要用自己的代码评估了
        # t0=time.time()
        gt_annos = self.prepare_test_gt(other_metrics)
        # print(time.time()-t0)

        from pycocotools.mask import encode
        # 将gt也转换result的形式，之后用format_results的方法转换成coco对象
        annos_list = gt_annos.copy()
        for i, anno in enumerate(annos_list):
            bbox_with_score = np.hstack([anno['gt_bboxes'], np.ones((len(anno['gt_bboxes']), 1))])
            annos_list[i] = (
                # [bbox_with_score], [encode(np.asfortranarray(anno['gt_masks'].masks.transpose(1, 2, 0)))])
                [bbox_with_score])

        dt_matches = None
        # 按照CocoDataset的输入形式将需要用coco评估的构造进coco_result
        if len(coco_metric) > 0:
            coco_results = list()
            for result in thred_results:
                coco_result = []
                for metric in coco_metric:
                    coco_result.append(result[metric])
                coco_results.append(tuple(coco_result))
                # bonai的data config虽然写了random_flip,flip_ratio=0.5,但是其实并没有进行翻转增广，这是因为pipline中的MultiScaleFlipAug会提前将
                # results['flip'] = False添加进数据流，从而测试中防止翻转增广，因此可以直接用数据集中保存的原始数据和最终的图片预测结果验证

            # anno={}
            # new_imgToAnns={}
            # m=1
            # from pycocotools.mask import encode
            # for k in range(len(gt_annos)):
            #     img_annos=[]
            #     for i, bbox in enumerate(gt_annos[k]['gt_bboxes']):
            #         img_annos.append({'area':(bbox[2]-bbox[0])*(bbox[3]-bbox[1]),'bbox':bbox,
            #                                       'category_id':1,'id':m})
            #         # pycocotools.mask.encode的输入若是H，W，C的三维度array，则输出是list of dict的多个rle，若是单个H,W的而唯独array，则输出单个rle
            #         # 其中一个rle包含{'size': ,'count': }两项
            #         anno[m]={'area':(bbox[2]-bbox[0])*(bbox[3]-bbox[1]),'bbox':bbox,'segmentation':encode(np.asfortranarray(gt_annos[0]['gt_masks'].masks[0])),
            #                         'image_id':k+1,'category_id':1,'iscrowd':0,'id':m}
            #         m+=1
            #     new_imgToAnns[k + 1] = img_annos
            # self.coco.anns=anno
            # self.coco.imgToAnns=new_imgToAnns

            # import matplotlib.pyplot as plt
            # mask_test1=np.zeros((1024,1024)).astype(np.uint8)
            # mask_test2=np.zeros((1024,1024)).astype(np.uint8)
            # mask_test1[200:600,200:600]=1
            # mask_test2[int(600-400*0.72):600,200:600]=1# iou=0.72
            # self.data_infos = self.data_infos[:1]
            # bbox_with_score = np.hstack([gt_annos[0]['gt_bboxes'][:1], np.ones((1, 1))])
            # # mask1 = np.asfortranarray(gt_annos[0]['gt_masks'].masks.transpose(1, 2, 0)[:, :, :1])
            # mask1 = np.asfortranarray(mask_test1)
            # plt.imshow(mask1)
            # plt.show()
            # annos_list1 = [([bbox_with_score], [encode(mask1[:,:,np.newaxis])])]
            # # mask2 = np.asfortranarray(gt_annos[0]['gt_masks'].masks.transpose(1, 2, 0)[:, :, 1:2])
            # mask2 = np.asfortranarray(mask_test2)
            # plt.imshow(mask2)
            # plt.show()
            # annos_list2 = [([bbox_with_score], [encode(mask2[:,:,np.newaxis])])]
            # self.data_infos=self.data_infos[:1]
            # eval_results, dt_matches = super(BONAI, self).evaluate(annos_list1, gt=annos_list2,
            #                                                        metric=coco_metric)

            eval_results, dt_matches,cocoEval = super(BONAI, self).evaluate(coco_results, gt=annos_list,
                                                                   metric=coco_metric)
        else:
            eval_results = {}
        eval_offset = False
        if ('offset' in metrics and 'offset' in results[0]) or ('segm' in metrics and 'segm' in results[0]):
            assign_gt_offset = []
            assign_pre_offset = []
            segm_iou_list = []
            segm_iou_merge_roof_list=[]
            match_id_start = 1  # dt_match是连续记录多张图片的bbox的，因此下一张图片的bbox编号要减去其起始值
            sum_iou_time = 0
            matched_polys_gt = []
            poly_match_to_target = []
            multi_poly_polis_scores = []
            MCAEs=[]
            ious_list=[]
            chamfer_dist2d_list=[]
            chamfer_dist3d_list=[]
            for i, result in tqdm(enumerate(results)):# 除了segm由于pack bits无法从result中提取，offset等都从thred_results中提取
                if result is None:
                    continue
                pre_offsets = thred_results[i]['offset']
                for j, pre_offset in enumerate(pre_offsets):
                    if j == len(dt_matches[
                                    i]):  # dt_matches只提了前1000个obj计算，而result总数（即pre_offsets总数）可能有多达2000个。用已经训练好的模型似乎result就会减少一些
                        break
                    if dt_matches[i][j] != 0:
                        assign_pre_offset.append(pre_offset)
                        # assign_gt_offset.append(gt_offsets[i][dt_match[j]-1])
                        assign_gt_offset.append(gt_annos[i]['gt_offsets'][dt_matches[i][j] - match_id_start])
                        if len(coco_results[i]) > 1:  # 有segm在coco_results中 ,即从single_gpu_test输出的rle类型的segm
                            segm_iou_list.append(mask.iou([coco_results[i][1][0][j]],
                                                          [annos_list[i][1][0][dt_matches[i][j] - match_id_start]],
                                                          [0]))

                dt_matches_np = np.array(dt_matches[i])
                vaild_mask = dt_matches_np != 0


                if vaild_mask.any():
                    if 'gt_segs' in gt_annos[i]:  # for off nadir multi class
                        gt_seg_target = get_instance_seg_with_indicator(gt_annos[i]['gt_segs'],
                                                                        gt_annos[i]['instance_indicator'],
                                                                        dt_matches_np[vaild_mask] - match_id_start)
                    else:
                        gt_seg_target = gt_annos[i]['gt_masks'][dt_matches_np[vaild_mask] - match_id_start]
                    time0 = time.time()
                    segm_unpack = np.unpackbits(result['segm'][0]['bits']).reshape(result['segm'][0]['shape'])# todo：免去unpack，使用直接根据bit类型计算iou的方法。后面的polygon可以只将需要的进行unpack。另外，将iou计算结果放入COCOeval统计coco相关参数
                    segm_unpack=segm_unpack[thred_results[i]['thre_ids']]
                    segm_match_to_target = segm_unpack[:len(dt_matches_np)][vaild_mask]
                    segm_iou = iou(segm_match_to_target,
                                   gt_seg_target)  # todo:background的iou是不正确的，gt中的background包含整个图像，应该在bbox范围内进行计算

                    ious = np.zeros((len(segm_unpack), len(gt_annos[i]['gt_masks'])))
                    for k, segm in enumerate(segm_unpack):
                        for j, gt_mask in enumerate(gt_annos[i]['gt_masks']):
                            if cocoEval.ious[(i, 1)][k, j] > 0:
                                ious[k, j] = iou(segm[None, :], gt_mask[None, :])
                            else:
                                ious[k, j] = 0
                    ious_list.append(ious)

                    if segm_unpack[0].ndim==2:
                        pre_roof_merge = np.sum(np.array(segm_unpack), axis=0)>0
                    elif segm_unpack[0].ndim==3:
                        pre_roof_merge = np.sum(np.array(segm_unpack)[:, 2], axis=0) > 0
                    gt_roof_merge = gt_annos[i]['gt_segs'][:, :,0]==2 if 'gt_segs' in gt_annos[i] else np.sum(gt_annos[i]['gt_masks'],axis=0)>0
                    segm_iou_merge_roof=iou(pre_roof_merge[None,...],gt_roof_merge[None,...])

                    # # iou可视化
                    # import matplotlib.pyplot as plt
                    # canva = np.zeros((1024, 1024, 3))
                    # canva[:, :, 0] = pre_roof_merge
                    # canva[:, :, 1] = gt_roof_merge
                    # plt.imshow(canva)
                    # plt.show()

                    sum_iou_time += time.time() - time0

                    if 'polygon' in metrics:
                        if 'polygons' in result:
                            pre_polygons = [result['polygons'][0][k]['tol_0.125'] for k in
                                            range(len(result['polygons'][0]))]
                        else:
                            polygonize_post_process = {
                                'method': 'simple',
                                "data_level": 0.5,
                                "tolerance": [0.125, 2],
                                "seg_threshold": 0.5,
                                "min_area": 10

                            }
                            from frame_field_learning.polygonize_simple import polygonize
                            if segm_match_to_target.ndim == 3:
                                pre_polygons_matched = polygonize(segm_match_to_target[:, None],
                                                                  # todo:在框内进行polygonize以提高速度
                                                                  config=polygonize_post_process)
                                segm_unpack = np.array(segm_unpack)[:,
                                                   None]  # channel: background, facade,roof
                            elif segm_match_to_target.ndim == 4:  # for off nadir multi class
                                pre_polygons_matched = polygonize(segm_match_to_target[:, 2:],
                                                                  config=polygonize_post_process)  # channel: background, facade,roof
                                segm_unpack = np.array(segm_unpack)[:, 2:]  # channel: background, facade,roof
                            vaild_gt_ids = np.array([gt_id for i, gt_id in enumerate(dt_matches_np[vaild_mask]) if
                                                     len(pre_polygons_matched[0][i]['tol_2']) > 0])
                            pre_polygons_matched = [pre_polygon_dict['tol_2'][0] for pre_polygon_dict in
                                                    pre_polygons_matched[0] if
                                                    len(pre_polygon_dict['tol_2']) > 0]  # 只用第一个
                            pre_polygons_thre = polygonize(segm_unpack,
                                                           config=polygonize_post_process)  # todo:上面已经矢量化过的可以直接拿过来
                            multi_polys_pre_thre = MultiPolygon(
                                [pre_polygon_dict['tol_2'][0] for pre_polygon_dict in pre_polygons_thre[0] if
                                 len(pre_polygon_dict['tol_2']) > 0])
                            # if 'crossfield' in result:
                            #     pass
                            #     # from frame_field_learning.polygonize_acm import polygonize as polygonize_acm
                            #     # polygon_batch = [[np.array(pre_polygon_dict['tol_0.125'][0].exterior.xy).transpose() for
                            #     #                   pre_polygon_dict in pre_polygons_thre[0] if
                            #     #                   len(pre_polygon_dict['tol_0.125']) > 0]]
                            #     # polygonize_acm_cfg = {
                            #     #     "steps": 500,
                            #     #     "data_level": 0.5,
                            #     #     # "data_coef": 0.1,
                            #     #     "data_coef": 0.0,
                            #     #     # "length_coef": 0.4,
                            #     #     "length_coef": 0.0,
                            #     #     "crossfield_coef": 0.5,
                            #     #     "poly_lr": 0.01,
                            #     #     "warmup_iters": 100,
                            #     #     "warmup_factor": 0.1,
                            #     #     "device": "cuda",
                            #     #     "tolerance": [0.125, 1],
                            #     #     "seg_threshold": 0.5,
                            #     #     "min_area": 10
                            #     # }
                            #     # # acm_result = polygonize_acm(torch.tensor(segm_thre_filter[0:1]),
                            #     # #                             torch.tensor(result['crossfield'][None, :]), polygonize_acm_cfg,
                            #     # #                             pre_computed={'init_contours_batch': polygon_batch})
                            #     # # acm_result = polygonize_acm(torch.tensor(segm_thre_filter), torch.tensor(
                            #     # #     result['crossfield'][None, :].repeat(35, axis=0)), polygonize_acm_cfg,
                            #     # #                             pre_computed={'init_contours_batch': [[polygon_single] for
                            #     # #                                                                   polygon_single in
                            #     # #                                                                   polygon_batch[0]]})
                            #     # acm_result = polygonize_acm(torch.tensor(np.sum(segm_thre_filter, axis=0)[None, :] > 0),
                            #     #                             torch.tensor(result['crossfield'][None, :]),
                            #     #                             polygonize_acm_cfg,
                            #     #                             pre_computed={'init_contours_batch': polygon_batch})
                            # else:
                            #     pass  # todo:polygon simple
                            multi_polys_gt = MultiPolygon(
                                [Polygon(mask[0].reshape((-1, 2))) for mask in gt_annos[i]['gt_polygons'].masks])

                            if 'gt_pitch_ratio' in gt_annos[i]:
                                map_scale=0.59716
                                chd = ChamferDistance()
                                z_pre = thred_results[i]['offset'][:, 0] /  gt_annos[i]['gt_pitch_ratio']
                                z_pre = torch.tensor(z_pre)
                                xyz_pres = []
                                xy_pres=[]
                                for k, single_poly in enumerate(multi_polys_pre_thre):
                                    xy_pre = torch.tensor(single_poly.exterior.xy).permute(1,0)
                                    xy_pres.append((xy_pre))
                                    ft_pre_xy=(xy_pre-thred_results[i]['offset'][k])*map_scale
                                    roof_xyz_pre = torch.cat((ft_pre_xy, z_pre[k].repeat(xy_pre.shape[0])[:,None]), dim=1)
                                    ft_xyz_pre = torch.cat((ft_pre_xy, torch.zeros((ft_pre_xy.shape[0],1))), dim=1)
                                    xyz_pres.append(torch.cat([roof_xyz_pre,ft_xyz_pre],dim=0))

                                z_gt = gt_annos[i]['gt_offsets'][:, 0] / gt_annos[i]['gt_pitch_ratio']
                                z_gt = torch.tensor(z_gt)
                                xyz_gts = []
                                xy_gts=[]
                                for k, single_poly in enumerate(multi_polys_gt):
                                    xy_gt = torch.tensor(single_poly.exterior.xy).permute(1,0)
                                    xy_gts.append(xy_gt)
                                    ft_gt_xy = (xy_gt - gt_annos[i]['gt_offsets'][k])*map_scale
                                    roof_xyz_gt = torch.cat((ft_gt_xy, z_gt[k].repeat(xy_gt.shape[0])[:, None]),
                                                             dim=1)
                                    ft_xyz_gt = torch.cat((ft_gt_xy, torch.zeros((ft_gt_xy.shape[0], 1))), dim=1)
                                    xyz_gts.append(torch.cat([roof_xyz_gt, ft_xyz_gt], dim=0))

                                xy_pre_tensor = torch.cat(xy_pres, dim=0)
                                xy_pre_tensor = torch.cat([xy_pre_tensor, torch.zeros((xy_pre_tensor.shape[0],1))], dim=1)
                                xy_gt_tensor = torch.cat(xy_gts, dim=0)
                                xy_gt_tensor = torch.cat([xy_gt_tensor, torch.zeros(( xy_gt_tensor.shape[0],1))], dim=1)
                                dist1, dist2 = chd(xy_gt_tensor[None, :],
                                                   xy_pre_tensor[None, :])
                                chamfer_dist2d_list.append(torch.sqrt(dist2).mean() + torch.sqrt(dist1).mean())

                                dist1, dist2 = chd(torch.cat(xyz_pres, dim=0)[None, :],
                                                   torch.cat(xyz_gts, dim=0)[None, :])
                                chamfer_dist3d_list.append(torch.sqrt(dist2).mean() + torch.sqrt(dist1).mean())

                            gt_polygons_matched=[multi_polys_gt[i] for i in vaild_gt_ids - match_id_start]
                            mean_cos_angle_error = compute_polygon_contour_measures(pre_polygons_matched,gt_polygons_matched,
                                                                       sampling_spacing=0.1, min_precision=0.5,
                                                                       max_stretch=2)
                            MCAEs.append(mean_cos_angle_error)
                            matched_polys_gt.extend(gt_polygons_matched)  # 按照match关系排序
                            poly_match_to_target.extend(pre_polygons_matched)
                            if len(multi_polys_gt) > 0 and len(multi_polys_pre_thre) > 0:
                                thre_polis=multi_polygon_polis_metric(multi_polys_gt,multi_polys_pre_thre)
                                multi_poly_polis_scores.append(thre_polis)  # 将一张图的多个多边形整合成一个multipolygon对象，和标签的multipolygon对象计算polis
                            # # 可视化：
                            # import matplotlib.pyplot as plt
                            # import shapely
                            # # thre:
                            # for single_bdry in multi_polys_pre_thre.boundary:
                            #     plt.plot(*single_bdry.coords.xy, color='blue')
                            # matched
                            # for single_bdry in pre_polygons_matched:
                            #     if isinstance(single_bdry.boundary, shapely.geometry.MultiLineString):
                            #         plt.plot(*single_bdry.boundary[0].xy, color='blue')
                            #     elif isinstance(single_bdry.boundary, shapely.geometry.LineString):
                            #         plt.plot(*single_bdry.boundary.xy, color='blue')
                            # # gt:
                            # for single_bdry in multi_polys_gt.boundary:
                            #     plt.plot(*single_bdry.coords.xy, color='red')
                            # roof_iou=segm_iou[2] if 'gt_segs' in gt_annos[i] else segm_iou
                            # plt.title('matched mcae:%.4f,thre polis:%.4f,matched iou:%.4f'%(mean_cos_angle_error,thre_polis,roof_iou))
                            # plt.show()
                else:
                    segm_iou = [np.nan, np.nan, np.nan] if 'gt_segs' in gt_annos[i] else np.nan
                    segm_iou_merge_roof=0
                    ious_list.append([])
                segm_iou_list.append(segm_iou)
                segm_iou_merge_roof_list.append(segm_iou_merge_roof)

                match_id_start += len(gt_annos[i]['gt_offsets'])

            if len(segm_iou_list) > 0:
                for i, ious in enumerate(ious_list):
                    cocoEval.ious[(i, 1)] = ious
                cocoEval.evalImgs = [
                    cocoEval.evaluateImg(imgId, catId, areaRng, 1000) for catId in [1]
                    for areaRng in [[0, 10000000000.0], [0, 1024], [1024, 9216], [9216, 10000000000.0]]
                    for imgId in range(300)]
                cocoEval.accumulate()
                cocoEval.summarize()
                coco_metric_names = {
                    'mAP': 0,
                    'mAP_50': 1,
                    'mAP_75': 2,
                    'mAP_s': 3,
                    'mAP_m': 4,
                    'mAP_l': 5,
                    'AR@100': 6,
                    'AR@300': 7,
                    'AR@1000': 8,
                    'AR_s@1000': 9,
                    'AR_m@1000': 10,
                    'AR_l@1000': 11
                }
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]
                metric='segm'
                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val

            if len(segm_iou_list) > 0:
                print('total unpack and iou cal time:', sum_iou_time)
                segm_iou = np.nanmean(segm_iou_list, axis=0)
                if isinstance(segm_iou,np.ndarray):
                    segm_iou = segm_iou[2]  # background,facade，roof
                eval_results['segm_mean_iou'] = segm_iou
                eval_results['segm_iou_merge_roof']=np.mean(segm_iou_merge_roof_list)

            if len(chamfer_dist2d_list)>0:
                eval_results['chamfer_dist2d_list'] = np.mean(chamfer_dist2d_list)
                eval_results['chamfer_dist3d_list'] = np.mean(chamfer_dist3d_list)

            if 'polygon' in metrics:
                if len(poly_match_to_target) > 0:
                    polis_score = polis_metric(matched_polys_gt, poly_match_to_target)  # 对应好的多边形一个对一个计算polis
                    eval_results['polis_score'] = polis_score
                if len(multi_poly_polis_scores) > 0:
                    eval_results['multi_poly_polis_score'] = np.mean(multi_poly_polis_scores)
                    eval_results['mean_cos_angle_error']=np.mean(MCAEs)

            offset_mea = mean_absolute_error(assign_pre_offset, assign_gt_offset)
            eval_results['offset_mea'] = offset_mea

        if 'angle' in metrics and 'angle' in results[0]:
            angles = [result['angle'] for result in results]
            gt_angles = [gt_anno['gt_angle'] for gt_anno in gt_annos]

            # 用xy方向的长度来计算角度误差，实验发现与直接用角度差计算是相同的。
            # pre_xys = [result['xy_dir'].cpu().numpy() for result in results]
            # gt_xys = [np.array([np.cos(angle), np.sin(angle)]) for angle in gt_angles]
            # angle_errors=self.get_angle_error(pre_xys,gt_xys)

            angle_errors = np.absolute(np.array(gt_angles - np.array(angles)))
            for i, error in enumerate(angle_errors):
                if error > np.pi:
                    angle_errors[i] = 2 * np.pi - angle_errors[i]  # 部分超过180度的转换到小于180的那边
            angle_mae = np.mean(angle_errors)
            # angle_mae=mean_absolute_error(gt_angles,angles)
            eval_results['angle_mae'] = angle_mae
        if 'crossfield' in metrics and 'crossfield' in results[0]:
            mean_diff_angles=[]
            for i,result in enumerate(results):
                pre_uv = c4c8_to_uv(torch.tensor(result['crossfield'][None,:]))
                diff = (torch.arctan(pre_uv[0, 0, 0] / pre_uv[0, 0, 1]) - gt_annos[i]['gt_crossfields']) % (np.pi / 2)
                mask = np.logical_not(gt_annos[i]['gt_crossfields'] == 0)
                masked_diff=mask*diff.numpy()
                masked_diff_to90=np.min(np.array([masked_diff% (np.pi / 2),-(masked_diff%(-np.pi/2))]),axis=0)
                # import matplotlib.pyplot as plt
                # plt.imshow(masked_diff_to90)
                # plt.colorbar()
                # plt.show()
                mean_diff = np.sum(masked_diff_to90 * mask) / np.sum(mask)
                mean_diff_angles.append(mean_diff/np.pi*180)
            eval_results['crossfield_align_error'] = np.nanmean(mean_diff_angles)
        # return eval_results, dt_matches
        return eval_results, dt_matches, gt_annos

    def get_angle_error(self, dir_pred, dir_gt):
        angle_errors = []
        for i in range(len(dir_pred)):
            ith_dir_pred = dir_pred[i]
            ith_dir_gt = dir_gt[i]
            ith_dir_pred /= np.linalg.norm(ith_dir_pred)
            ith_dir_gt /= np.linalg.norm(ith_dir_gt)
            cos_ang = np.dot(ith_dir_pred, ith_dir_gt)  # 用点乘和叉乘构造三角函数差公式计算角度差
            sin_ang = np.linalg.norm(np.cross(ith_dir_pred, ith_dir_gt))
            rad_diff = np.arctan2(sin_ang, cos_ang)
            angle_error = np.degrees(rad_diff)
            angle_errors.append(angle_error)
        return angle_errors

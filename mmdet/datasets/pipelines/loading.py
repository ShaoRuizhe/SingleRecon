# encoding:utf-8
import math
import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 channel_order='bgr'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.channel_order=channel_order

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type,channel_order=self.channel_order)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadDataFromTorchPT(object):

    def __init__(self,to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        if results['img_info']['filename'].endswith('.png'):
            results['img_info']['filename']=results['img_info']['filename'][:-4]+'.pt'
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        data=torch.load(filename)
        img = data.get('image',None)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['gt_segs'] = data.get('gt_polygon_image',None)
        results['instance_indicator'] = data.get('instance_indicator',None)
        results['gt_crossfields'] = data.get('gt_crossfield_angle',None)/255*np.pi
        results['img_fields'] = ['gt_segs','gt_crossfields','instance_indicator','img']# 必须保证img是在最后一个的，否则resize过程中修改result['img_shape']可能导致与img的尺寸不一致（例如变成gt_crossfields的尺寸（1024，1024）而不是img的尺寸（1024，1024，3））
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadImageFromWebcam(LoadImageFromFile):
    """Load an image from webcam.

    Similar with :obj:`LoadImageFromFile`, but the image read from webcam is in
    ``results['img']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class LoadMultiChannelImageFromFiles(object):
    """Load multi-channel images from a list of separate channel files.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename", which is expected to be a list of filenames).
    Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load multiple images and get images meta
        information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = [
                osp.join(results['img_prefix'], fname)
                for fname in results['img_info']['filename']
            ]
        else:
            filename = results['img_info']['filename']

        img = []
        for name in filename:
            img_bytes = self.file_client.get(name)
            img.append(mmcv.imfrombytes(img_bytes, flag=self.color_type))
        img = np.stack(img, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load mutiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 with_offset=False,
                 with_building_height=False,
                 with_angle=False,
                 with_pitch_ratio=False,
                 with_rbbox=False,
                 with_edge=False,
                 with_side_face=False,
                 with_offset_field=False,
                 with_roof_bbox=False,
                 with_footprint_bbox=False,
                 with_only_footprint_flag=False,
                 with_roof_mask=False,
                 with_footprint_mask=False,
                 poly2mask=True,
                 file_client_args=dict(backend='disk')):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.with_offset = with_offset
        self.with_building_height = with_building_height
        self.with_angle = with_angle
        self.with_pitch_ratio=with_pitch_ratio
        self.with_rbbox = with_rbbox
        self.with_edge = with_edge
        self.with_side_face = with_side_face
        self.with_offset_field = with_offset_field
        self.with_roof_bbox = with_roof_bbox
        self.with_footprint_bbox = with_footprint_bbox
        self.poly2mask = poly2mask
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.with_only_footprint_flag = with_only_footprint_flag
        self.with_roof_mask = with_roof_mask
        self.with_footprint_mask = with_footprint_mask

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_roof_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_roof_bboxes'] = ann_info['roof_bboxes'].copy()

        results['bbox_fields'].append('gt_roof_bboxes')
        return results

    def _load_footprint_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_footprint_bboxes'] = ann_info['footprint_bboxes'].copy()

        results['bbox_fields'].append('gt_footprint_bboxes')
        return results

    def _load_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            results['gt_masks'] = BitmapMasks(
                    [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        results['gt_polygons'] =  PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        results['mask_fields'].append('gt_masks')
        results['mask_fields'].append('gt_polygons')
        return results

    def _load_roof_masks(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results['img_info']['height'], results['img_info']['width']
        gt_roof_masks = results['ann_info']['roof_masks']
        if self.poly2mask:
            gt_roof_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_roof_masks], h, w)
        else:
            gt_roof_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_roof_masks], h,
                w)
        results['gt_roof_masks'] = gt_roof_masks
        results['mask_fields'].append('gt_roof_masks')
        return results

    def _load_footprint_masks(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results['img_info']['height'], results['img_info']['width']
        gt_footprint_masks = results['ann_info']['footprint_masks']
        if self.poly2mask:
            gt_footprint_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_footprint_masks], h, w)
        else:
            gt_footprint_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_footprint_masks], h,
                w)
        results['gt_footprint_masks'] = gt_footprint_masks
        results['mask_fields'].append('gt_footprint_masks')
        return results

    def _load_semantic_seg(self, results):
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        results['gt_semantic_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def _load_offsets(self, results):
        """loading offset value

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded offset annotations.
        """
        ann_info = results['ann_info']
        results['gt_offsets'] = ann_info['offsets']
        results['offset_fields'].append('gt_offsets')
        return results

    def _load_building_heights(self, results):
        """loading building height value

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded height annotations.
        """
        ann_info = results['ann_info']
        results['gt_building_heights'] = ann_info['building_heights']
        return results

    def _load_pitch_ratio(self, results):
        ann_info = results['ann_info']
        results['gt_pitch_ratio'] = 2.5 if np.isnan(ann_info['pitch_ratio']) else ann_info['pitch_ratio']
        return results

    def _load_angle(self, results):
        """loading angle value
            若标注中有offsets，则计算图中所有建筑物offset角度的加权均值：average(offset_angles,wieghts=offset_lens)，作为输出的angle,否则直接使用标注中的angle
            23.11.02:标注中的angle其实是竖直方向的角度，计算方法是arctan（offset长度/高度与之），计算过程位于_parse_ann_info内
        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded angle annotations.
        """
        ann_info = results['ann_info']
        if ann_info.get('offsets',None) is not None:
            offsets = ann_info['offsets']
            offset=offsets.sum(0)
            angle=np.angle(offset[0]+offset[1]*1j)# 利用复数的实部虚部形式转换到角度模长形式获取角度
            results['gt_angle']=angle
            # is_0_idx = offsets[:, 0] == 0
            # offsets_angles = np.zeros_like(offsets[:, 0])
            # offsets_angles[is_0_idx] = np.pi / 2 * np.where(offsets[is_0_idx, 1] > 0, 1, -1)
            # offsets_angles[~is_0_idx] = np.vectorize(math.atan)(offsets[~is_0_idx, 1] / offsets[~is_0_idx, 0])
            # offsets_angles[~is_0_idx] += np.where(offsets[~is_0_idx][:, 0] > 0, 0, np.pi)
            # len_weight = np.square(offsets)  # 长度越长的，在求均值的时候权重越大
            # len_weight = len_weight[:, 0] + len_weight[:, 1]
            # results['gt_angle']=np.average(offsets_angles, weights=len_weight).astype(float)

        else:
            results['gt_angle'] = ann_info['angle']
        return results

    def _load_only_footprint_flag(self, results):
        """loading footprint flag which used in semi-supervised learning framework

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded footprint flag annotations.
        """
        ann_info = results['ann_info']
        results['gt_only_footprint_flag'] = ann_info['only_footprint_flag']
        return results

    def _load_rbboxes(self, results):
        ann_info = results['ann_info']
        results['gt_rbboxes'] = ann_info['rbboxes']
        results['rbbox_fields'].append('gt_rbboxes')
        return results

    def _load_edge_map(self, results):
        """loading the edge map which generated by weijia

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded edge map annotations.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['edge_prefix'],
                            results['ann_info']['edge_map'])
        img_bytes = self.file_client.get(filename)
        edge_maps = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()

        h, w = results['img_info']['height'], results['img_info']['width']
        mask_num = len(results['ann_info']['masks'])
        gt_edge_maps = BitmapMasks(
                [edge_maps for _ in range(mask_num)], h, w)

        results['gt_edge_maps'] = gt_edge_maps
        results['edge_fields'].append('gt_edge_maps')
        return results

    def _load_side_face_map(self, results):
        """loading side face map

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded side face map annotations.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['side_face_prefix'],
                            results['ann_info']['side_face_map'])
        img_bytes = self.file_client.get(filename)
        side_face_maps = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()

        h, w = results['img_info']['height'], results['img_info']['width']
        mask_num = len(results['ann_info']['masks'])
        gt_side_face_maps = BitmapMasks(
                [side_face_maps for _ in range(mask_num)], h, w)

        results['gt_side_face_maps'] = gt_side_face_maps
        results['side_face_fields'].append('gt_side_face_maps')
        return results

    def _load_offset_field(self, results):
        """loading offset field map which generated by weijia and lingxuan

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded offset field annotations.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['offset_field_prefix'],
                            results['ann_info']['offset_field'])

        gt_offset_field = np.load(filename).astype(np.float32)

        ignores_x, ignores_y = [], []
        for subclass in [400, 500]:
            ignores_x.append(gt_offset_field[..., 0] == subclass)
            ignores_y.append(gt_offset_field[..., 1] == subclass)

        ignore_x_bool = np.logical_or.reduce(tuple(ignores_x))
        ignore_y_bool = np.logical_or.reduce(tuple(ignores_y))

        gt_offset_field[..., 0][ignore_x_bool] = 0.0
        gt_offset_field[..., 1][ignore_y_bool] = 0.0

        results['gt_offset_field'] = gt_offset_field
        results['offset_field_fields'].append('gt_offset_field')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        if self.with_offset:
            results = self._load_offsets(results)
        if self.with_building_height:
            results = self._load_building_heights(results)
        if self.with_angle:
            results = self._load_angle(results)
        if self.with_pitch_ratio:
            results = self._load_pitch_ratio(results)
        if self.with_rbbox:
            results = self._load_rbboxes(results)
        if self.with_edge:
            results = self._load_edge_map(results)
        if self.with_side_face:
            results = self._load_side_face_map(results)
        if self.with_roof_bbox:
            results = self._load_roof_bboxes(results)
        if self.with_footprint_bbox:
            results = self._load_footprint_bboxes(results)
        if self.with_offset_field:
            results = self._load_offset_field(results)
        if self.with_only_footprint_flag:
            results = self._load_only_footprint_flag(results)
        if self.with_roof_mask:
            results = self._load_roof_masks(results)
        if self.with_footprint_mask:
            results = self._load_footprint_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg})'
        repr_str += f'poly2mask={self.poly2mask})'
        repr_str += f'poly2mask={self.file_client_args})'
        return repr_str


@PIPELINES.register_module()
class LoadProposals(object):
    """Load proposal pipeline.

    Required key is "proposals". Updated keys are "proposals", "bbox_fields".

    Args:
        num_max_proposals (int, optional): Maximum number of proposals to load.
            If not specified, all proposals will be loaded.
    """

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        """Call function to load proposals from file.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded proposal annotations.
        """

        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                f'but found {proposals.shape}')
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(num_max_proposals={self.num_max_proposals})'

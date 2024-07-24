# encoding:utf-8
# anno+img=>rotated anno+img
import json
import math
import os.path

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt
from mmcv import Config
from pycocotools.coco import COCO
from mmdet.core import BitmapMasks

# 以下函数get_angle_from_offset,offset_coordinate_transform,offset_rotate,get_corners,bbox_rotate,rotate_with_anno来自transforms.py
from mmdet.models import build_detector


def get_angle_from_offset(ann_info):
    assert ann_info[0].get('offset', None) is not None
    offsets = [ann_info_obj['offset'] for ann_info_obj in ann_info]
    offset = np.sum(offsets,0)
    angle = np.angle(offset[0] + offset[1] * 1j)  # 利用复数的实部虚部形式转换到角度模长形式获取角度
    return angle

def get_pitch(ann_info):
    assert ann_info[0].get('offset', None) is not None
    assert ann_info[0].get('building_height', None) is not None
    offsets = [ann_info_obj['offset'] for ann_info_obj in ann_info]
    building_heights = [ann_info_obj['building_height'] for ann_info_obj in ann_info]
    building_heights=np.array(building_heights)
    if sum(building_heights==0):
        print('!')
        return np.nan
    offsets=np.array(offsets)[np.logical_not(building_heights==0)]
    # building_heights[building_heights==0]=3
    pitch_ratio = np.sum(np.linalg.norm(offsets,axis=1), 0)/np.sum(building_heights)
    pitch_ratio=np.clip(pitch_ratio,1.5,4.)
    return pitch_ratio

def offset_coordinate_transform(offset, transform_flag='xy2la'):
    """transform the coordinate of offsets

    Args:
        offset (list): list of offset
        transform_flag (str, optional): flag of transform. Defaults to 'xy2la'.

    Raises:
        NotImplementedError: [description]

    Returns:
        list: transformed offsets
    """
    if transform_flag == 'xy2la':
        offset_x, offset_y = offset
        length = math.sqrt(offset_x ** 2 + offset_y ** 2)
        angle = math.atan2(offset_y, offset_x)
        offset = [length, angle]
    elif transform_flag == 'la2xy':
        length, angle = offset
        offset_x = length * np.cos(angle)
        offset_y = length * np.sin(angle)
        offset = [offset_x, offset_y]
    else:
        raise NotImplementedError

    return offset

def offset_rotate(offset,rotate_angle):
    offset = offset_coordinate_transform(offset, transform_flag='xy2la')
    offset = [offset[0], offset[1] + rotate_angle * math.pi / 180.0]
    offset = offset_coordinate_transform(offset, transform_flag='la2xy')
    return np.array(offset, dtype=np.float32)

def get_corners(bboxes):
    """Get corners of bounding boxes

    Parameters
    ----------

    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners

def bbox_rotate(bboxes, img_shape, rotate_angle):
    """rotate bboxes.

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    if bboxes.shape[0] == 0:
        return bboxes
    corners = get_corners(bboxes)
    corners = np.hstack((corners, bboxes[:, 4:]))

    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))
    angle = rotate_angle
    h, w, _ = img_shape
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    calculated = np.dot(M, corners.T).T
    calculated = np.array(calculated, dtype=np.float32)
    calculated = calculated.reshape(-1, 8)

    x_ = calculated[:, [0, 2, 4, 6]]
    y_ = calculated[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    rotated = np.hstack((xmin, ymin, xmax, ymax))
    return rotated

def poly_rotate(seg_polys,M):
    # 这里的旋转形式为中心旋转 返回为2维array形式的poly，这里不会进行出界判断
    if np.array(seg_polys).shape[0]==1 or len(np.array(seg_polys).shape)==1:
        seg_polys = np.array(seg_polys).reshape((-1, 2))
    for j in range(len(seg_polys)):
        seg_polys[j] = M.dot([seg_polys[j, 0], seg_polys[j, 1], 1])
    return seg_polys

def rotate_with_anno(anno,img,angle,auto_bound=False):
    # 对img,offset,segmentation,bbox进行旋转，其余的舍弃
    # auto_bound: 若True，根据旋转自动调整图像大小；若False，以图像中心点为中心旋转，不调整大小，使用原图像大小
    angle = angle * 180 / np.pi
    # offset原始计算角度为顺时针为正，mmcv.imrotate的旋转也是顺时针(本质上是cv取负，mmcv再取负，详见mmcv.imrotate代码），
    # 因此要通过旋转将offset归零，就要mmcv.imrotate(-gt_angle)
    img_shp=img.shape
    rotated_img = mmcv.imrotate(
        img, angle,center=(img_shp[0] / 2,img_shp[1] / 2), auto_bound=auto_bound)
    M = cv2.getRotationMatrix2D((img_shp[0] / 2, img_shp[1] / 2),-angle, 1)
    # 关于角度方向：cv2.getRotationMatrix2D计算矩阵时将角度取负，结果时对图像的逆时针旋转，
    # mmcv.imrotate本质上是角度取负，在交给cv2.getRotationMatrix2D，结果时对图像的顺时针旋转
    # 这里对于bbox的旋转方向应当与mmcv.imrotate一致，因此为负的mmcv.imrotate角度，也即offset原始计算角度本身
    if auto_bound:  # 需要在旋转后增加一个平移，平移距离等于图像因旋转而增加的尺寸的一半
        M[0, 2] += (img_shp[0] - img_shp[0]) / 2
        M[1, 2] += (img_shp[1] - img_shp[1]) / 2
    rotated_anno=[]
    for anno_obj in anno:
        rotated_anno_obj={}
        # rotate offsets
        offset=rotated_anno_obj['offset'] = offset_rotate(anno_obj['offset'], angle)
        # rotate seg poly
        poly=poly_rotate(anno_obj['segmentation'],M)
        rotated_anno_obj['segmentation'] = poly.reshape(-1)
        # 根据roof polygon和offset获取新的bbox
        left = min(poly[:, 0])
        top = min(poly[:, 1])
        right = max(poly[:, 0])
        bottom = max(poly[:, 1])
        if offset[0] > 0:
            left -= offset[0]
        else:
            right -= offset[0]
        if offset[1] > 0:
            top -= offset[1]
        else:
            bottom -= offset[1]
        if left < 1024 and top < 1024 and right > 0 and bottom > 0:  # 转出去的就不要了 todo：开启auto bound的情况图像尺寸不为1024？
            rotated_anno_obj['building_bbox']=([left, top, right-left, bottom-top])
            rotated_anno.append(rotated_anno_obj)
    return rotated_anno,rotated_img

def show_anno(anno,img):
    cfg = Config.fromfile('configs/loft_foa/loft_foa_r50_fpn_2x_bonai.py')
    gt_bbox = np.array(
        [np.append(anno_obj['building_bbox'], 1) for anno_obj in anno])
    for i in range(len(gt_bbox)):
        x,y,w,h,c=gt_bbox[i]
        gt_bbox[i]=[x,y,x+w,y+h,c] # 读取数据过程中的BONAI._frase_ann_info中会将标注中的box以x,y,w,h的形式读入，并通过x,y,x+w,y+h转化为x1,y1,x2,y2形式
    gt_result = {'bbox': gt_bbox}
    cfg.model.neck=None
    cfg.model.rpn_head=None
    cfg.model.roi_head=None
    model=build_detector(cfg.model)
    model.show_result(img, gt_result, score_thr=0, num_obj=100)

def make_rotated_data(split='trainval'):# trainval/test
    cfg = Config.fromfile('configs/loft_foa/loft_foa_r50_fpn_2x_bonai_server_92.py')
    out_dir=r'/datapool/data/BONAI/rotated_with_pitch/'+split
    os.makedirs(out_dir,exist_ok=True)
    if split=='train':
        img_path=cfg.data.train.img_prefix
        anno_path=cfg.data.train.ann_file
    elif split=='test':
        img_path=cfg.data.val.img_prefix
        anno_path=cfg.data.test.ann_file
    imgToAnns_cat=[]
    imgs=[]
    # for json_file in anno_path:
    coco_annos=COCO(anno_path)
    imgToAnns_cat.extend(coco_annos.imgToAnns.values())
    imgs.extend(coco_annos.imgs.values())
    anno_dicts=[]
    for i,img in enumerate(imgs):
        img_npy=mmcv.imread(mmcv.imread(os.path.join(img_path,img['file_name'])))
        angle=get_angle_from_offset(imgToAnns_cat[i])
        pitch_ratio=get_pitch(imgToAnns_cat[i])
        anno,rotated_img=rotate_with_anno(imgToAnns_cat[i],img_npy,-angle)# offset计算的角度顺时针为正，旋转方向正角度顺时针旋转，因此这里要取负
        anno_dict={'anno':anno,'pitch_ratio':pitch_ratio}
        anno_dict['file_name']=img['file_name']
        anno_dict['angle']=angle
        anno_dicts.append(anno_dict)
        # show_anno(anno,rotated_img)
        # mmcv.imwrite(rotated_img,os.path.join(out_dir,'images',img['file_name']))
        pass
    # 整理anno为coco形式
    json_coco={'annotations':[],'images':[],'categories':[{'id':1,'name':'building'}]}# 'categories':[{'id':1,'name':'building'}]
    anno_id=0
    for img_id,anno_dict in enumerate(anno_dicts):
        for anno in anno_dict['anno']:
            anno['id']=anno_id
            anno_id+=1
            anno['image_id']=img_id
            anno['category_id']=1
            anno['iscrowd']=0.0
            anno['angle']=anno_dict['angle']
            anno['pitch_ratio']=anno_dict['pitch_ratio']
            json_coco['annotations'].append(anno)
        image={'file_name':anno_dict['file_name']}
        image['id']=img_id
        image['width']=1024
        image['height']=1024
        json_coco['images'].append(image)
    mmcv.dump(json_coco,os.path.join(out_dir,'annotation.json'))


if __name__ == '__main__':
    # anno+img=>rotated anno+img
    make_rotated_data('test')
    # make_rotated_data('train')
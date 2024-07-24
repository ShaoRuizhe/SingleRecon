# coding=UTF-8
from mmcv import Config
from pycocotools.coco import COCO
import cv2
import numpy as np

def COCOSegmAnnotation2Mask(file_names,output_path):
    # 从file_names中给的coco json文件中读取图像与roof-mask的匹配关系，然后将roof-mask转换到mask形式（1-roof 0-其他），存储到output_path/image_name.png(image_name即图像的文件名)
    if not isinstance(file_names,list):
        file_names=[file_names]
    for file_name in file_names:
        coco_anno=COCO(file_name)
        for img in coco_anno.imgs.values():
            mask=np.zeros((img['height'],img['width']),np.uint8)
            for anno in coco_anno.imgToAnns[img['id']]:
                mask=cv2.bitwise_or(mask,coco_anno.annToMask(anno))
                # print(anno['id'])
            cv2.imwrite(output_path+img['file_name'][:-4]+'.png',(mask*255).astype(np.uint8))# 不能使用jpg，否则可能有奇怪的插值，有254或是1这种




if __name__ == '__main__':
    cfg=Config.fromfile('../../configs/loft_foa/loft_foa_r50_fpn_2x_bonai.py')
    COCOSegmAnnotation2Mask(cfg.test_ann_file,cfg.data_root+'roof_mask/test/')
    # COCOSegmAnnotation2Mask(cfg.train_ann_file,cfg.data_root+'roof_mask/')
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet.datasets.pipelines import to_tensor
from mmdet.models import build_detector
def BONAI_building_detection(img,img_meta,show=False):
    cfg = Config.fromfile('configs/loft_foa/loft_foa_r50_fpn_2x_bonai_data_rotate.py')
    checkpoint=r'D:\Documents\PycharmProjects\building_footprint\BONAI\work_dir\test_rotate_0905\epoch_20.pth'
    norm_cfg=cfg.img_norm_cfg
    img_float=mmcv.imnormalize(img, std=np.array(norm_cfg.std),mean=np.array(norm_cfg.mean),to_rgb=norm_cfg.to_rgb)
    img_tensor=to_tensor(img_float.transpose(2, 0, 1))
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.eval()
    with torch.no_grad():
        result = model([img_tensor[None,...]],[[ img_meta]],return_loss=False, rescale=True)# 这么多层list啥的都是哪里加进来的？
    if show:
        model.show_result(img,result[0],score_thr=0.3,num_obj=50)
    return result
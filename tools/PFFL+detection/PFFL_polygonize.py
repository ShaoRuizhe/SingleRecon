import json

import numpy as np
import torch
from torch_lydorn import torchvision
from jsmin import jsmin
from frame_field_learning import data_transforms, polygonize
from backbone import get_backbone
from frame_field_learning.model import FrameFieldModelOffNadir
from lydorn_utils import run_utils

def PFFL_polygonize(target_segs):
    config_filepath = r'PFFL_configs\polygonize_params.json'
    num_buildings=40
    with open(config_filepath, 'r') as f:
        minified = jsmin(f.read())
        config = json.loads(minified)
    selected_polygons=[]
    selected_prob=[]
    for i in range(num_buildings):
        polygons_batch,probs_batch=polygonize.polygonize(config, target_segs["interior_seg"][i][2][None, None, ...],
                              crossfield_batch=target_segs['crossfield'][i][None, ...])
        if len(polygons_batch[0])==0 or len(polygons_batch[0]['acm'].get('tol_1'))==0:# 前项代表compute_init_contours_batch未检测出多边形，后者代表init有多边形，可以进入后续方法，但是后续方法经过优化没有输出多边形
            selected_polygons.append(None)
            continue
        areas=[polygon.area for polygon in polygons_batch[0]['acm']['tol_1']]
        biggest_id=np.argmax(areas)
        if probs_batch[0]['acm']['tol_1'][biggest_id]>probs_batch[0]['acm']['tol_0.125'][biggest_id]:
            selected_polygons.append(polygons_batch[0]['acm']['tol_1'][biggest_id])
            selected_prob.append(probs_batch[0]['acm']['tol_1'][biggest_id])
        else:
            selected_polygons.append(polygons_batch[0]['acm']['tol_0.125'][biggest_id])
            selected_prob.append(probs_batch[0]['acm']['tol_0.125'][biggest_id])
    return selected_polygons,probs_batch

def PFFL_seg(img):
    checkpoint_file= r'D:\Documents\PycharmProjects\building_detection/PFFL-BONAI_exp/coor_horiz_loss/checkpoint.best_val.epoch_000076.tar'
    config_filepath=r'PFFL_configs\config.bonai_rotated.unet_resnet101_pretrained.json'
    with open(config_filepath, 'r') as f:
        minified = jsmin(f.read())
        config = json.loads(minified)
    run_utils.load_defaults_in_config(config, filepath_key="defaults_filepath")
    eval_online_cuda_transform = data_transforms.get_eval_online_cuda_transform(config)
    train_online_cuda_transform = data_transforms.get_online_cuda_transform(config,
                                                                    augmentations=config["data_aug_params"]["enable"])
    backbone = get_backbone(config["backbone_params"])
    model = FrameFieldModelOffNadir(config, backbone=backbone, train_transform=train_online_cuda_transform,
                                    eval_transform=eval_online_cuda_transform)
    model.load_state_dict(torch.load(checkpoint_file)['model_state_dict'])
    model.eval()

    # ref:PFFL::inference_from_filepath.py.inference_from_filepath
    image_float = img / 255
    mean = np.mean(image_float.reshape(-1, image_float.shape[-1]), axis=0)
    std = np.std(image_float.reshape(-1, image_float.shape[-1]), axis=0)
    sample = {
        "image": torchvision.transforms.functional.to_tensor(img)[None, ...],
        # 注意这里是自己写的torch_lydorn里面的to_tensor,并不会将图像转换到0-1的float。。。
        "image_mean": torch.from_numpy(mean)[None, ...],
        "image_std": torch.from_numpy(std)[None, ...],
    }

    with torch.no_grad():
        pred, batch = model(sample)
    return pred
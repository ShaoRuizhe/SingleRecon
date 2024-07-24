# encoding:utf-8
import argparse
import json
import os
import time

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from pathlib import Path
import PIL

from mmdet.datasets.pipelines import to_tensor
from mmdet.core import wrap_fp16_model
from mmdet.models import build_detector
from tools.to_3d import save_3dbuilding
from tools.py3dtilers.GeojsonTiler.GeojsonTiler import GeojsonTiler
from frame_field_learning.polygonize_simple import polygonize

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('model_config')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--data-config',default='configs/_base_/datasets/bonai_data_rotate_lenovo_polygon_eval.py')
    parser.add_argument('--out')
    parser.add_argument('--image')
    parser.add_argument('--color-type',default='rgb',choices=['rgb','bgr'])
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--no-crossfield', action='store_true')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.5,
        help='score threshold (default: 0.5)')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use ')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    assert args.image is not None or args.image_ids is not None

    model_cfg = Config.fromfile(args.model_config)
    data_cfg = Config.fromfile(args.data_config)
    # set cudnn_benchmark
    if model_cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    model_cfg.model.pretrained = None
    if model_cfg.model.get('neck'):
        if model_cfg.model.neck.get('rfp_backbone'):
            if model_cfg.model.neck.rfp_backbone.get('pretrained'):
                model_cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        model_cfg.gpu_ids = args.gpu_ids
    else:
        model_cfg.gpu_ids = [0]
    torch.cuda.set_device(model_cfg.gpu_ids[0])

    meta = [[{'img_shape': (1024, 1024, 3), 'scale_factor': np.array([1, 1, 1, 1]), 'ori_shape': (1024, 1024, 3),
              'flip':False,'flip_direction':None}]]
    map_scale = 0.59  # meter per pixel
    # build the model and load checkpoint
    model = build_detector(model_cfg.model, train_cfg=None, test_cfg=model_cfg.test_cfg)
    model.eval()
    fp16_cfg = model_cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.out is None:
        args.out=r'D:\Documents\PycharmProjects\building_footprint\BONAI\work_dir\3d_output'
    img_paths = []
    pitch_ratios = []
    geojson_out_dirs = []
    tile_out_dirs = []
    if args.image is not None:
        img_paths.append(args.image)
        pitch_ratios.append(2.5)# offset(pixel)/pitch_ratio=height(meter)
        geojson_out_dirs.append(os.path.join(args.out,Path(args.image).stem,'geojson'))
        tile_out_dirs.append(os.path.join(args.out,Path(args.image).stem,'3dtiles'))

    times=[]
    for img_path, pitch_ratio,geojson_out_dir,tile_out_dir in zip(img_paths,pitch_ratios,geojson_out_dirs,tile_out_dirs):
        time0=time.time()
        img=np.array(PIL.Image.open(img_path))
        if args.color_type=='bgr':
            img_bgr=img[:,:,::-1]
            img_tensor=mmcv.imnormalize(img_bgr, np.array(data_cfg.img_norm_cfg['mean']), np.array(data_cfg.img_norm_cfg['std']),
                                        np.array(data_cfg.img_norm_cfg['to_rgb']))
        else:
            img_tensor = mmcv.imnormalize(img, np.array(data_cfg.img_norm_cfg['mean']),
                                          np.array(data_cfg.img_norm_cfg['std']),
                                          np.array(data_cfg.img_norm_cfg['to_rgb']))
        img_tensor=to_tensor(img_tensor.transpose(2, 0, 1))


        with torch.no_grad():
            result = model(return_loss=False, rescale=True, img=[img_tensor[None,:]],img_metas=meta)
        thre_mask=result[0]['bbox'][0][:,4]>args.score_thr
        result = {'bbox': result[0]['bbox'][0][thre_mask],
                  'segm': [np.array(result[0]['segm'][0])[thre_mask]],
                  'offset': result[0]['offset'][thre_mask]}

        if args.show:
            model.show_result(
                img,
                result,
                show=True,score_thr=args.score_thr,out_file=args.show_dir)

        os.makedirs(geojson_out_dir, exist_ok=True)
        polygon_roof = polygonize(result['segm'][0][:, None,:],data_cfg.polygonize_post_process)
        height_scale=1/(pitch_ratio*map_scale)
        building_dicts=[{'roof_mask':ply['tol_2'][0],'offset':off} for ply,off in zip(polygon_roof[0],result['offset']) if len(ply['tol_2'])>0]
        save_3dbuilding(img=img, building_dicts=building_dicts,
                        outdir=geojson_out_dir,height_scale=height_scale)

        os.makedirs(tile_out_dir, exist_ok=True)
        geojson_tiler = GeojsonTiler()
        geojson_tiler.parse_command_line(
            # path=[r'D:\Documents\PycharmProjects\building_footprint\BONAI\work_dir\3d_geojson\buildings.geojson'],
            # path=[geojson_out_dir + '\\base_map.geojson'],
            path=[geojson_out_dir+'\\buildings.geojson',
                 geojson_out_dir+'\\base_map.geojson'],
            out_dir=tile_out_dir)
        properties = ['height', 'height',
                      'width', 'LARGEUR',
                      'prec', 'PREC_ALTI',
                      'z', 'NONE']
        tileset = geojson_tiler.from_geojson_directory(properties)
        if tileset is not None:
            print("Writing tileset in",tile_out_dir)
            tileset.write_as_json(tile_out_dir)
        times.append(time.time()-time0)
    print(times)
    print(np.mean(times))

if __name__ == '__main__':
    main()

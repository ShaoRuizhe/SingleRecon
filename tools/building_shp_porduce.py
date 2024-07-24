# encoding:utf-8
import argparse
import os
import time

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from osgeo import ogr, osr
from tqdm import tqdm

from tools.fuse_conv_bn import fuse_module

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def TransPixCor2GeoCor(PixCor,GeoTrans):
    X = GeoTrans[0] + PixCor[0] * GeoTrans[1] + PixCor[1] * GeoTrans[2];
    Y = GeoTrans[3] + PixCor[0] * GeoTrans[4] + PixCor[1] * GeoTrans[5];
    return [X,Y]

def TransCtrPix2Geo(ctr,GeoTrans):
    # 将pix坐标的轮廓转换为地理坐标
    # ctr的格式：n_polygon,1,n_vertex,2
    ctr_geo=[]
    for polygon in ctr:
        polygon=np.array(polygon).squeeze()
        polygon_geo=[]
        for vertex in polygon:
            vtx_geo=TransPixCor2GeoCor(vertex,GeoTrans)
            polygon_geo.append(vtx_geo)
        ctr_geo.append(polygon_geo)
    return ctr_geo

def Vectorize2Shp(masks,offsets,height_ratio,geo_trans,sr,output_path,lbl_mask=None):
    """

    Args:
        masks: 输入的网络预测mask，注意，与之前CDproc中不同的是，这里的输入有多个mask，每个mask对应一个polygon，而无需将mask判断连通区域，每个连通区域一个polygon
        offsets:
        height_ratio:建筑物高度height=norm(offset)*height_ratio
        geo_trans: 6要素的geo trans
        sr: 投影坐标系
        output_path: 输出shp文件名（包含路径）
        lbl_mask:
    Returns:
        不返回值，会将根据mask生成的地理矢量存储到shpfile中
    """
    # 输入坐标系信息与完整预测图，输出shp矢量文件
    # contours_changed=vectorize_coarse(img_input)# 存在多边形被丢失
    # contours_changed=vectorize_fine(img_input)# 存在边缘的多边形提取错误
    area_thre=100
    masks = (masks > 0).astype(np.uint8)
    lbl_mask=lbl_mask.astype(np.uint8) if lbl_mask is not None else None

    polygon_areas = []
    intersect_with_lable = []
    polygons = []
    for j,mask in enumerate(masks):
        offset=offsets[j]
        ret, binary = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)  # 需要的img数据类型是uint8，否则cv2会报错
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # shape = img_input.shape[:2]
        # contours_image = draw(contours,shape)
        # plt.imshow(contours_image)
        # plt.show()

        # 关于ring和polygon： https://www.osgeo.cn/python_gdal_utah_tutorial/ch03.html

        # 创建新polygon对象
        polygon = ogr.Geometry(ogr.wkbPolygon)
        # 处理多边形孔洞的情形：
        # 由cv2得到的多边形时通过hierarchy来记录当前contour是否是上一个contour的内环的
        # geometry的polygon对象是通过子对象wkbLinearRing来保存内环和外环的
        # 这里就是要将cv2的多边形记录方式转化为geometry的记录方式，从而实现有孔洞的地理多边形对象
        polygon_area=0.0
        # print(len(contours))
        # 第一个多边形的范围，即外围轮廓的范围，即第一个轮廓对象的范围
        x, y, w, h = cv2.boundingRect(contours[0])
        for i,cnt in tqdm(enumerate(contours)):
            # time0=time.time()
            cnt_area = cv2.contourArea(cnt)
            if cnt_area >=area_thre:
                epsilon = 0.001 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                # 创建ring对象
                ring = ogr.Geometry(ogr.wkbLinearRing)
                # 将顶点一个个添加进ring里面去
                for vertex in approx:
                    ring.AddPoint(*TransPixCor2GeoCor(vertex[0]-offset,geo_trans))
                # 对于当前contour的结算都要放到metr判断当前contour是否是孔洞以及对上一个polygon的结算后
                polygon_area+= cnt_area*(1 if hierarchy[0,i,3]==-1 else -1)
                # 将ring添加到polygon
                polygon.AddGeometry(ring)
        polygons.append(polygon)
        polygon_areas.append(polygon_area)
        if lbl_mask is not None:
            # if (mask*lbl_mask).any():
            if (cv2.bitwise_and(mask[y:y + h, x:x + w], lbl_mask[y:y + h, x:x + w])).any():
                intersect_with_lable.append(True)
            else:
                intersect_with_lable.append(False)



    # 创建shp文件
    source_ds = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(
        utf8_path=output_path)
    # 创建Driver的later
    source_lyr = source_ds.CreateLayer('poly', srs=sr, geom_type=ogr.wkbPolygon)
    # 为这个layer添加ID属性
    source_lyr.CreateField(ogr.FieldDefn('ID', ogr.OFTInteger))
    source_lyr.CreateField(ogr.FieldDefn('area', ogr.OFTReal))
    source_lyr.CreateField(ogr.FieldDefn('height', ogr.OFTReal))
    for i in range(len(polygons)):
        # 创建feature
        feat = ogr.Feature(source_lyr.GetLayerDefn())
        # 将polygons设置为feature的Geom
        feat.SetGeometryDirectly(polygons[i])
        # 设置feature的ID
        feat.SetField('ID', i)
        feat.SetField('area', polygon_areas[i])
        feat.SetField('height',float(np.linalg.norm(offsets[i])*height_ratio))
        if len(intersect_with_lable)==len(polygon_areas):
            feat.SetField('corr', int(intersect_with_lable[i]))
        # 将这个feature加入layer中
        source_lyr.CreateFeature(feat)

def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
        score_thre=0.5
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
                bbox_score= result['bbox'][0][:,4]
                mask_results=np.array(result['segm'][0])[bbox_score>score_thre]
                offset_results=result['offset'][bbox_score>score_thre]
                sr=osr.SpatialReference()
                sr.ImportFromEPSG(3857)
                geo_trans=(-7.90e+06,0.15,0,5.205e+06,0,-0.15)
                Vectorize2Shp(mask_results, offset_results, 0.5, geo_trans, sr,
                              output_path=r'C:\Users\srz\Desktop\bonai_test\bonai_test_{}.shp'.format(i))
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.options is None else args.options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            dataset.evaluate(outputs, args.eval, **kwargs)


if __name__ == '__main__':
    main()

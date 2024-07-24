# encoding:utf-8
import argparse
import os

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import pycocotools.mask as mask_util
from tools.fuse_conv_bn import fuse_module
import matplotlib.pyplot as plt

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model, tensor2imgs
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from segment_anything import sam_model_registry, SamPredictor

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
        # outputs ,mask_results= single_gpu_test(model, data_loader, args.show, args.show_dir,
        #                           args.show_score_thr)
        outputs= single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    img=tensor2imgs(dataset[0]['img'][0][None,:,:,:], **dataset[0]['img_metas'][0]._data['img_norm_cfg'])
    print("======> Load SAM" )
    args.ckpt=r'D:\Documents\PycharmProjects\cv开源项目\segment-anything-main\ckpt\sam_vit_h_4b8939.pth'
    sam_type, sam_ckpt = 'vit_h', args.ckpt
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    predictor = SamPredictor(sam)

    # Image feature encoding
    i =9
    left = int(outputs[0]['bbox'][0][i, 0])
    top = int(outputs[0]['bbox'][0][i, 1])
    right = int(outputs[0]['bbox'][0][i, 2])
    bottom = int(outputs[0]['bbox'][0][i, 3])
    plt.imshow(img[0][top:bottom, left:right])
    plt.show()

    predictor.set_image(img[0])
    # predictor.set_image(img[0][top:bottom, left:right])# 如果要测试bbox输入到sam的效果，将此行接触注释，并将上一行注释，然后将下面的 bboxed_segm_bitmap部分取消注释
    bbox_size=[bottom-top,right-left]
    test_feat = predictor.features.squeeze()

    # Positive-negative location prior
    if isinstance(outputs[0]['segm'][0][0],dict):# rle形式的mask
        segm_bitmap=mask_util.decode(outputs[0]['segm'][0][0])# pycocotools.mask.decode的输入是dict类型的rle紧凑型mask{'size':list[2],'counts':bytes} 三个0分别是第0张图，第0类，第0个框
    else:
        segm_bitmap = outputs[0]['segm'][0][i]# 没有经过阈值的模型响应值，ndarray
        bboxed_segm_bitmap=segm_bitmap[top:bottom, left:right]
    topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(torch.tensor(segm_bitmap), topk=1)
    # topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(torch.tensor(bboxed_segm_bitmap), topk=1)

    # topk_xy_i[0,1]+=100

    topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
    topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)
    # per-sam中，原图的大小是1k+,而输入mask的大小是predict输出的low_res_mask,尺寸为256，这样才能让输入mask的predict过程中图片token和mask prompt的token形状相同可以相加
    mask_input=cv2.resize(segm_bitmap,(256,256))[None,:,:]# mask的编码过程将mask缩小了4倍，图像的编码都是到固定的64，因此mask应当都是4*64=256*256的
    # mask_input=cv2.resize(bboxed_segm_bitmap,(256,256))[None,:,:]
        # Cascaded Post-refinement-1
    masks, scores, logits= predictor.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        mask_input=mask_input,
        multimask_output=True)
    best_idx = np.argmax(scores)

    for mask in masks:
        plt.imshow(mask[top:bottom, left:right])
        # plt.imshow(mask)
        plt.show()

    return# 后面的没调，可能报错

    # Cascaded Post-refinement-2
    y, x = np.nonzero(masks[best_idx])
    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()
    input_box = np.array([x_min, y_min, x_max, y_max])
    masks, scores, logits = predictor.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        box=input_box[None, :],
        mask_input=logits[best_idx: best_idx + 1, :, :],
        multimask_output=True)
    best_idx = np.argmax(scores)

    # 以下是eval过程
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.options is None else args.options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_results,dt_match=dataset.evaluate(outputs, args.eval, **kwargs)



    bbox_match=[]
    segm_match=[]
    offset_match=[]
    # 只对第一张图进行可视化测试
    for i in range(outputs[0]['bbox'][0].shape[0]):
        if dt_match[0][i]!=0:# 匹配到非背景
            bbox_match.append(outputs[0]['bbox'][0][i])
            # segm_match.append(mask_results[0][i])
            offset_match.append(outputs[0]['offset'][i])
        # else:
        #     if i<100:# 将匹配到背景的框显示出来看
        #     #     import matplotlib.pyplot as plt
        #     #
        #     #     plt.imshow(img[0])
        #     #     bbox = outputs[0]['bbox'][0][i]
        #     #     ax = plt.gca()
        #     #     ax.add_patch(
        #     #     plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], color="blue", fill=False,
        #     #                           linewidth=1))
        #     #     bbox = list(dataset.coco.anns.values())[13]['building_bbox']
        #     #     ax.add_patch(
        #     #     plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], color="blue", fill=False,
        #     #                           linewidth=1))
        #     #     plt.show()
        #         print(i)

    # result_match = {'bbox': [np.array(bbox_match)], 'segm': [(segm_match)], 'offset': np.array(offset_match)}
    result_match = {'bbox': [np.array(bbox_match)], 'offset': np.array(offset_match)}
    model.module.show_result(img[0],result_match,score_thr=0,num_obj=100)# 不设置thre，都显示出来

    gt_bbox=np.array([np.append(anno['building_bbox'],1) for anno in list(dataset.coco.anns.values()) if anno['image_id']==1])
    # gt_bbox=np.array([np.append(anno['bbox'],1) for anno in list(dataset.coco.anns.values()) if anno['image_id']==1])
    gt_bbox[:, 2], gt_bbox[:, 3] = gt_bbox[:, 2] + gt_bbox[:, 0], gt_bbox[:, 3] + gt_bbox[:, 1]
    gt_result={'bbox':[gt_bbox]}# 将anns中的真值bbox转化为show_result输入需要的result形式
    model.module.show_result(img[0], gt_result, score_thr=0,num_obj=100 )
    print('done')


def point_selection(mask_sim, topk=1):
    # ref:per-sam
    # Top-1 point selection
    h, w = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_y = (topk_xy // w).unsqueeze(0)
    topk_x = (topk_xy - topk_y * w)
    topk_xy = torch.cat((topk_x, topk_y), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()

    thre=0.95
    pos_points=torch.where(mask_sim>thre)
    topk_xy=np.array([[pos_points[1].float().mean(),pos_points[0].float().mean()]])# 只取一个正类的中心点作为pos point prompt。注意顺序，从行，列->x,y要交换顺序

    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()

    return topk_xy, topk_label, last_xy, last_label

if __name__ == '__main__':
    main()

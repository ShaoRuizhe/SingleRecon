# encoding:utf-8
import argparse
import os

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from tools.fuse_conv_bn import fuse_module
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model, tensor2imgs
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
# from tools.chamfer_distance import ChamferDistance # C:\Users\srz\AppData\Local\torch_extensions\torch_extensions\Cache\py39_cu117\cd

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
        '--show-interval', type=int,default=200,help='directory where painted images will be saved')
    parser.add_argument(
        '--no-crossfield', action='store_true')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.5,
        help='score threshold (default: 0.5)')
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
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use ')
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

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids=[0]
    torch.cuda.set_device(cfg.gpu_ids[0])
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
        model = MMDataParallel(model, device_ids=[torch.cuda.current_device()])
        # outputs ,mask_results= single_gpu_test(model, data_loader, args.show, args.show_dir,
        #                           args.show_score_thr)
        outputs= single_gpu_test(model, data_loader, args.show, args.show_dir,args.show_interval,args.no_crossfield,
                                  args.show_score_thr)
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
        if args.eval is None:
            args.eval=cfg.evaluation.metric
            # eval_results,dt_match=dataset.evaluate(outputs, args.eval, **kwargs)
        eval_results,dt_match,gt_annos=dataset.evaluate(outputs, args.eval, **kwargs)
        for key,value in eval_results.items():
            print(key,':',value)
    tensor_imgs = torch.cat([data['img'][0][None, :] for data in dataset], dim=0)
    imgs = tensor2imgs(tensor_imgs, **dataset[0]['img_metas'][0]._data['img_norm_cfg'])

    bbox_match=[]
    segm_match=[]
    offset_match=[]

    # 显示第一章图的 展示gt
    from pycocotools.mask import decode
    # matched_ids = np.unique(dt_match[0])
    # matched_ids = matched_ids[matched_ids > 0] - 1
    for i in range(len(gt_annos)):
        gt_bbox = np.hstack([gt_annos[i]['gt_bboxes'], np.ones((len(gt_annos[i]['gt_bboxes']), 1))])
        gt_segm = np.asfortranarray(gt_annos[i]['gt_masks'])
        gt_result = {'bbox': gt_bbox, 'segm': [
            gt_segm], 'offset': gt_annos[i][
            'gt_offsets']}  # 原gt_annos[i]中有两项，第一项是bbox，第二项是经过rle编码的mask。但把poly放在gt_mask处之后就不有第二项了，因此此处的gt_annos[0][1]会报错。
        # model.module.show_result(imgs[i], gt_result, score_thr=0, num_obj=100, show=True)
        model.module.show_result(imgs[i], gt_result, score_thr=0, num_obj=200,
                                 out_file='../work_dir/gt_imgs/%d.jpg' % i)

    if dt_match is not None:# 对于目标检测模型的可视化
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
        # 显示匹配到的预测结果
        # result_match = {'bbox': [np.array(bbox_match)], 'segm': [(segm_match)], 'offset': np.array(offset_match)}
        result_match = {'bbox': [np.array(bbox_match)], 'offset': np.array(offset_match)}# 不会处理rel类型的mask（字符串紧凑表示）
        model.module.show_result(imgs[0], result_match, score_thr=0, num_obj=100)# 不设置thre，都显示出来



    else: # 对于角度预测模型的可视化：
        for i in range(len(dataset)):
            imgs = tensor2imgs(dataset[i]['img'][0][None, :, :, :], **dataset[i]['img_metas'][0]._data['img_norm_cfg'])
            model.module.show_result(imgs[0], outputs[i])
            model.module.show_result(imgs[0], {'angle':dataset[i]['gt_angle'][0]})# 显示标签的角度

if __name__ == '__main__':
    main()

# encoding:utf-8
import os
import os.path as osp
import pickle
import shutil
import tempfile
import time

import cv2
import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
import numpy as np

from mmdet.core import encode_mask_results, tensor2imgs
from tools.to_3d import save_3dbuilding


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_interval=200,
                    no_crossfield=False,
                    show_score_thr=0.3,
                    polygonize_method=None):
    """
        用model测试dataloader中的数据，每隔show_interval个batch显示一次，显示是会将batch中的多个图片绘制出来并在上面展示结果

    Args:
        model:
        data_loader:
        show:
        out_dir:
        show_interval:
        show_score_thr:

    Returns:
        results：list[num_data] of result_dict，其中每个dict与模型simple_test的输出一致。对于CrossfieldMultiScale模型，result_dict为
            dict{'bbox','segm','offset',('crossfield')}
                'segm':list[3] 其中3是三那个类别，只有第一个类别存储了数据。'segm'[0]:dict('len':len,'bits':ndarray[len*3*1024*1024/8])，这是一个经过packbits的array，pack之后有效地节约了内存。
            原尺寸为len，3，1024，1024，可以通过np.unpackbits(np.packbits(segm)).reshape((len，3,1024,1024))恢复到原本的array。
            其中，len为目标数量，3为3个类别（background,facade,roof),1024是图像尺寸
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    if out_dir is not None:
        os.makedirs(out_dir[:out_dir.rfind('/')], exist_ok=True)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        show_flag = (i % show_interval == 0)
        # if out_dir is not None and not show_flag:  # for eval hook calling
        #     continue

        # 真值标签3d化
        # if i in [59, 61, 64, 84, 99, 152, 199, 275]:
        # if i in [ 99,]:
        #     vis_img = tensor2imgs(data['img'][0], **data['img_metas'][0].data[0][0]['img_norm_cfg'])[0][:, :, ::-1]
        #     map_scale=0.59
        #     for obj_anno in dataset.coco.anns.values():
        #         if obj_anno['image_id'] == i:
        #             if not np.isnan(obj_anno['pitch_ratio']):
        #                 pitch_ratio=obj_anno['pitch_ratio']
        #             else:
        #                 pitch_ratio=2.5
        #             break
        #     height_scale = 1 / (pitch_ratio * map_scale)
        #     building_dicts = [{'offset': obj_ann['offset'], 'roof_mask': obj_ann['segmentation']} for obj_ann
        #                       in dataset.coco.imgToAnns[i]]
        #     save_3dbuilding(img=vis_img,
        #                     building_dicts=building_dicts,
        #                     outdir=r'D:\Documents\PycharmProjects\building_footprint\BONAI\work_dir\3d_output\gt\img_num%d'%i,
        #                     height_scale=height_scale)
        # continue

        with torch.no_grad():
            # time0=time.time()
            result = model(return_loss=False, rescale=True, **data) # list[batch] of dict{'bbox','segm','offset',('crossfield'),('polygons')}
            # print('ref time:',time.time()-time0)

        # 模型输出3d化
        # mask = result[0]['segm'][0][6]
        # ret, binary = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)  # 需要的img数据类型是uint8，否则cv2会报错
        # contours, hierarchy = cv2.findContours(binary.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        #
        # save_3dbuilding(img=vis_img, building_dicts=[{'roof_mask':contours[0][:,0,:],'offset':result[0]['offset'][6]}],
        #                 outdir=r'C:\Users\srz\Desktop\RS img examples\building_set',height_scale=0.2)

        if show or out_dir:
            if no_crossfield:
                if 'crossfield' in result[0]:
                    result[0].pop('crossfield')
            if show_flag:
                img_tensor = data['img'][0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for img, img_meta,result_single_img in zip(imgs, img_metas,result):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        # out_file = osp.join(out_dir,img_meta['ori_filename'])
                        out_file = out_dir+'-'+img_meta['ori_filename']
                        out_file = out_file[:out_file.rfind('.')]+'.jpg'
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result_single_img,
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)

        if len(result[0]['segm'][0][0].shape) == 2:# orgin model: single channel
            for single_img_result in result:
                mask_results = np.array(single_img_result.get('segm', None)[0])
                # single_img_result['segm'] = encode_mask_results(mask_results)
                # mask_results = np.round(single_img_result.get('segm', None)[0]).astype(bool)
                single_img_result['segm'] = [{'shape': mask_results.shape, 'bits': np.packbits(mask_results)}]
        elif len(result[0]['segm'][0][0].shape) == 3:# offnadir model: multi channels
            for single_img_result in result:
                mask_results = np.array(single_img_result.get('segm', None)[0])# channel:background,facade,roof # todo:also process facade
                # single_img_result['segm'] = encode_mask_results(mask_results[:,2:])
                single_img_result['segm'] =[{'shape':mask_results.shape,'bits':np.packbits(mask_results)}]
        results += result

        # if isinstance(result,list):# 有batch的two_stage输出方式：list[batch_size](dict{'bbox'......})
        #     if 'segm' in result[0] and isinstance(result[0]['segm'][0],list):# 若没有segm输出或segm输出为packbite，则跳过rle编码
        #         for single_img_result in result:
        #             single_img_result['segm'][0] = {'len': len(single_img_result['segm'][0]),
        #                                       'bits': np.packbits(np.array(single_img_result['segm'][0]))}
        #             if len(single_img_result['segm'][0][0].shape)<3:
        #                 mask_results = single_img_result.get('segm', None)
        #                 # if mask_results is not None:# 如果是二值mask，就转换成紧凑的rle类型存储
        #                     ## segm不需要一定是bool，encode_mask_results中有归置到uint8的，然后才会encode
        #                     # single_img_result['segm'] = encode_mask_results(mask_results)# list[2000|num instances] of dict{'size:[1024,1024],'counts':bytes}
        #             else:# offnadir的三类（background，facade，roof）就不要encode了
        #                 pass
        #
        # else:# 其他未更新的输出方式仍然按照旧的来
        #     results.append(result)
        #
        batch_size = len(data['img_metas'][0].data[0])
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result, tuple) and len(result) == 2:
                bbox_results, mask_results = result
                encoded_mask_results = encode_mask_results(mask_results)
                result = bbox_results, encoded_mask_results
            elif isinstance(result, tuple) and len(result) == 3:
                bbox_results, mask_results, offset_results = result
                if mask_results is not None:
                    encoded_mask_results = encode_mask_results(mask_results)
                    result = bbox_results, encoded_mask_results, offset_results
                else:
                    # only pred offset
                    result = bbox_results, offset_results
            elif isinstance(result, tuple) and len(result) == 4:
                bbox_results, mask_results, offset_results, height_results = result
                encoded_mask_results = encode_mask_results(mask_results)
                result = bbox_results, encoded_mask_results, offset_results, height_results
        results.append(result)

        if rank == 0:
            batch_size = (
                len(data['img_meta'].data)
                if 'img_meta' in data else len(data['img_metas'][0].data))
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results

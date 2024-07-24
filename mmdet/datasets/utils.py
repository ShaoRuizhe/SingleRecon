# --coding:utf-8--
import numpy as np


def iou(pr, gt, eps=1e-7):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (Tensor or ndarray): [n,c,h,w]
        gt (Tensor or ndarray):  [n,c,h,w]
        eps (float): epsilon to avoid zero division
    Returns:
        float: IoU (Jaccard) score
    """
    gt=np.array(gt)
    assert pr.ndim == gt.ndim and pr.shape == gt.shape
    IoU_result = []
    if pr.ndim==3:
        intersection = np.sum(gt * pr)
        union = np.sum(gt) + np.sum(pr) - intersection + eps
    elif pr.ndim==4:
        intersection = np.sum(gt * pr, axis=(0, 2, 3))
        union = np.sum(gt, axis=(0, 2, 3)) + np.sum(pr, axis=(0, 2, 3)) - intersection + eps
    return (intersection + eps) / union


def get_instance_seg_with_indicator(gt_seg, instance_indicator, dt_matches):
    """
        get instance seg with indicator only for single image
    Args:
        gt_seg:ndarray[h,w,3]:
        instance_indicator:ndarray[h,w]
        dt_matches:list/ndarray[n_pre_instances]
    output:
        array[n_pre_instances,n_class,h,w]
    """
    indicator_masks = (
            instance_indicator[:, :, None].repeat(len(dt_matches),axis=2) == np.array(dt_matches))
    gt_masks_th = indicator_masks * gt_seg.transpose(2, 0, 1)[..., None]  # borcast:indicator_masks[h,w,n_pre_instances] gt_seg.permute(2, 0, 1)[..., None]:  [3,h,w,1]-> gt_masks_th[3,h,w,n_pre_instances]
    gt_masks_th = gt_masks_th.transpose(0, 3, 1, 2)# [3,n_pre_instances,h,w,]
    gt_masks_interior_th = array2one_hot(gt_masks_th[0], 3).transpose(0, 3, 1, 2)  # background,roof,facade 3 classses
    # gt_masks_seg_th = array2one_hot(gt_masks_th[1], 4).permute(0, 3, 1,
    #                                                            2)  # background,roof_right,between,facade_up_down 3 classses
    # gt_masks_vertex_th = array2one_hot(gt_masks_th[2], 2).permute(0, 3, 1, 2)  # background,vertex 2 classses
    return gt_masks_interior_th# [n_pre_instances,n_class,h,w]


def array2one_hot(input: np.ndarray, n_classes: int):
    """

    Args:
        input  :  ndarray[n,h,w]
    output:
        ndarray[n,h,w,c]
    """
    assert input.max() < n_classes
    one_hot_array = np.zeros(np.append(input.shape, n_classes), dtype=bool)# [n,h,w,c]
    for c in range(n_classes):
        one_hot_array[..., c][input == c] = True
    return one_hot_array

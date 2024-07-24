# encoding:utf-8
import math
from functools import partial

import scipy.interpolate
import numpy as np
import torch
import torch.distributed
from torch.nn import functional as F

from . import measures
from . import frame_field_utils

import torch_lydorn.kornia

from lydorn_utils import math_utils, print_utils


# --- Base classes --- #


class Loss(torch.nn.Module):
    def __init__(self, name):
        """
        Attribute extra_info can be used in self.compute() to add intermediary results of loss computation for
        visualization for example.
        It is the second output of self.__call__()

        :param name:
        """
        super(Loss, self).__init__()
        self.name = name
        self.norm_meter = None
        self.norm = torch.nn.parameter.Parameter(torch.Tensor(1), requires_grad=False)
        self.reset_norm()
        self.extra_info = {}  #

    def reset_norm(self):
        self.norm_meter = math_utils.AverageMeter("{}_norm".format(self.name), init_val=1)
        self.norm[0] = self.norm_meter.val

    def update_norm(self, pred_batch, gt_batch, nums):
        loss = self.compute(pred_batch, gt_batch)
        self.norm_meter.update(loss, nums)
        self.norm[0] = self.norm_meter.val

    def sync(self, world_size):
        """
        This method should be used to synchronize loss norms across GPUs when using distributed training
        :return:
        """
        torch.distributed.all_reduce(self.norm)
        self.norm /= world_size

    def compute(self, pred_batch, gt_batch):
        raise NotImplementedError

    def forward(self, pred_batch, gt_batch, normalize=True):
        loss = self.compute(pred_batch, gt_batch)
        if normalize:
            assert 1e-9 < self.norm[0], "self.norm[0] <= 1e-9 -> this might lead to numerical instabilities."
            loss = loss / self.norm[0]
        extra_info = self.extra_info
        self.extra_info = {}  # Re-init extra_info
        # contains_nan = bool(torch.sum(torch.isnan(loss)).item())
        # assert not contains_nan, f"loss {str(self)} is Nan!"
        return loss, extra_info

    def __repr__(self):
        return "{} (name={}, norm={:0.06})".format(self.__class__.__name__, self.name, self.norm[0])


class MultiLoss(torch.nn.Module):
    def __init__(self, loss_funcs, weights, epoch_thresholds=None, pre_processes=None):
        """

        @param loss_funcs:
        @param weights:
        @param pre_processes: List of functions to call with 2 arguments (which are updated): pred_batch, gt_batch to compute only one values used by several losses.
        """
        super(MultiLoss, self).__init__()
        assert len(loss_funcs) == len(weights), \
            "Should have the same amount of loss_funcs ({}) and weights ({})".format(len(loss_funcs), len(weights))
        self.loss_funcs = torch.nn.ModuleList(loss_funcs)

        self.weights = []
        for weight in weights:
            if isinstance(weight, list):
                # Weight is a list of coefs corresponding to epoch_thresholds, they will be interpolated in-between
                self.weights.append(scipy.interpolate.interp1d(epoch_thresholds, weight, bounds_error=False, fill_value=(weight[0], weight[-1])))
            elif isinstance(weight, float) or isinstance(weight, int):
                self.weights.append(float(weight))
            else:
                raise TypeError(f"Type {type(weight)} not supported as a loss coef weight.")

        self.pre_processes = pre_processes

        for loss_func, weight in zip(self.loss_funcs, self.weights):
            if weight == 0:
                print_utils.print_info(f"INFO: loss '{loss_func.name}' has a weight of zero and thus won't affect grad update.")

    def reset_norm(self):
        for loss_func in self.loss_funcs:
            loss_func.reset_norm()

    def update_norm(self, pred_batch, gt_batch, nums):
        if self.pre_processes is not None:
            for pre_process in self.pre_processes:
                pred_batch, gt_batch = pre_process(pred_batch, gt_batch)
        for loss_func in self.loss_funcs:
            loss_func.update_norm(pred_batch, gt_batch, nums)

    def sync(self, world_size):
        """
        This method should be used to synchronize loss norms across GPUs when using distributed training
        :return:
        """
        for loss_func in self.loss_funcs:
            loss_func.sync(world_size)

    def forward(self, pred_batch, gt_batch, normalize=True, epoch=None):
        if self.pre_processes is not None:
            for pre_process in self.pre_processes:
                pred_batch, gt_batch = pre_process(pred_batch, gt_batch)
        total_loss = 0
        # total_weight = 0
        individual_losses_dict = {}
        extra_dict = {}
        for loss_func_i, weight_i in zip(self.loss_funcs, self.weights):
            loss_i, extra_dict_i = loss_func_i(pred_batch, gt_batch, normalize=normalize)
            if isinstance(weight_i, scipy.interpolate.interpolate.interp1d) and epoch is not None:
                current_weight = float(weight_i(epoch))
            else:
                current_weight = weight_i
            total_loss += current_weight * loss_i
            # total_weight += weight_i
            individual_losses_dict[loss_func_i.name] = loss_i
            extra_dict[loss_func_i.name] = extra_dict_i
        # total_loss /= total_weight
        return total_loss, individual_losses_dict, extra_dict

    def __repr__(self):
        ret = "\n\t".join([str(loss_func) for loss_func in self.loss_funcs])
        return "{}:\n\t{}".format(self.__class__.__name__, ret)


# --- Build combined loss: --- #
def compute_seg_loss_weigths(pred_batch, gt_batch, config):
    """
    Combines distances (from U-Net paper) with sizes (from https://github.com/neptune-ai/open-solution-mapping-challenge).

    @param pred_batch:
    @param gt_batch:
    @return:
    """
    device = gt_batch["image"].device
    use_dist = config["loss_params"]["seg_loss_params"]["use_dist"]
    use_size = config["loss_params"]["seg_loss_params"]["use_size"]
    w0 = config["loss_params"]["seg_loss_params"]["w0"]
    sigma = config["loss_params"]["seg_loss_params"]["sigma"]
    height = gt_batch["image"].shape[2]
    width = gt_batch["image"].shape[3]
    im_radius = math.sqrt(height * width) / 2

    # --- Class imbalance weight (not forgetting background):
    if config['loss_params'].get('losses_set_name') == "off_nadir":
        channel_class=['interior','edge','vertex']
        freq_weights={}
        for i,class_freq in enumerate(gt_batch['class_freq'][0]):# 默认batch中各条数据的class_freq是一样的，直接用第一条数据集的class_freq
            if config['seg_params']['compute_' + channel_class[i]]:
                if isinstance(class_freq,list):
                    pass# todo:统计下每个类别的频率
                else:
                    num_classes=config['seg_params']['compute_'+channel_class[i]]
                    backgroudn_freq=1-class_freq
                    pixel_class_freq=[backgroudn_freq]+[class_freq/(num_classes-1)]*(num_classes-1)
                    pixel_class_freq=torch.tensor(pixel_class_freq).to(device)
                freq_weights[channel_class[i]] = 1 / pixel_class_freq
        gt_batch["seg_loss_weights"] = freq_weights
        return pred_batch,gt_batch

    gt_polygons_mask = (0 < gt_batch["gt_polygons_image"]).float()
    background_freq = 1 - torch.sum(gt_batch["class_freq"], dim=1)
    pixel_class_freq = gt_polygons_mask * gt_batch["class_freq"][:, :, None, None] + \
                       (1 - gt_polygons_mask) * background_freq[:, None, None, None]
    if pixel_class_freq.min() == 0:
        print_utils.print_error("ERROR: pixel_class_freq has some zero values, can't divide by zero!")
        raise ZeroDivisionError
    freq_weights = 1 / pixel_class_freq
    # print("freq_weights:", freq_weights.min().item(), freq_weights.max().item())

    # Compute size weights
    # print("sizes:", gt_batch["sizes"].min().item(), gt_batch["sizes"].max().item())
    # print("distances:", gt_batch["distances"].min().item(), gt_batch["distances"].max().item())
    # print("im_radius:", im_radius)
    size_weights = None
    if use_size:
        if gt_batch["sizes"].min() == 0:
            print_utils.print_error(("ERROR: sizes tensor has zero values, can't divide by zero!"))
            raise ZeroDivisionError
        size_weights = 1 + 1 / (im_radius * gt_batch["sizes"])

    distance_weights = None
    if use_dist:
        # print("distances:", gt_batch["distances"].min().item(), gt_batch["distances"].max().item())
        distance_weights = gt_batch["distances"] * (height + width)  # Denormalize distances
        distance_weights = w0 * torch.exp(-(distance_weights ** 2) / (sigma ** 2))
        # print("sum(distances == 0):", torch.sum(gt_batch["distances"] == 0).item())
        # print("distance_weights:", distance_weights.min().item(), distance_weights.max().item())

        # print(distance_weights.shape, distance_weights.min().item(), distance_weights.max().item())
        # print(size_weights.shape, size_weights.min().item(), size_weights.max().item())
        # print(freq_weights.shape, freq_weights.min().item(), freq_weights.max().item())

    gt_batch["seg_loss_weights"] = freq_weights
    if use_dist:
        gt_batch["seg_loss_weights"] += distance_weights
    if use_size:
        gt_batch["seg_loss_weights"] *= size_weights

    # print(gt_batch["seg_loss_weights"].shape, gt_batch["seg_loss_weights"].min().item(), gt_batch["seg_loss_weights"].max().item())
    # print("seg_loss_weights:", size_weights.min().item(), size_weights.max().item())

    # print("freq_weights:", freq_weights.min().item(), freq_weights.max().item())
    # print("size_weights:", size_weights.min().item(), size_weights.max().item())
    # print("distance_weights:", distance_weights.min().item(), distance_weights.max().item())

    # Display:
    # display_seg_loss_weights = gt_batch["seg_loss_weights"][0].cpu().detach().numpy()
    # display_distance_weights = distance_weights[0].cpu().detach().numpy()
    # skimage.io.imsave("seg_loss_dist_weights.png", display_distance_weights[0])
    # display_size_weights = size_weights[0].cpu().detach().numpy()
    # skimage.io.imsave("seg_loss_size_weights.png", display_size_weights[0])
    # display_freq_weights = freq_weights[0].cpu().detach().numpy()
    # display_freq_weights = display_freq_weights - display_freq_weights.min()
    # display_freq_weights /= display_freq_weights.max()
    # skimage.io.imsave("seg_loss_freq_weights.png", np.moveaxis(display_freq_weights, 0, -1))
    # for i in range(3):
    #     skimage.io.imsave(f"seg_loss_weights_{i}.png", display_seg_loss_weights[i])
        # skimage.io.imsave(f"freq_weights_{i}.png", display_freq_weights[i])

    return pred_batch, gt_batch


def compute_gt_field(pred_batch, gt_batch):
    gt_crossfield_angle = gt_batch["gt_crossfield_angle"]
    gt_field = torch.cat([torch.cos(gt_crossfield_angle),
                   torch.sin(gt_crossfield_angle)], dim=1)
    gt_batch["gt_field"] = gt_field
    return pred_batch, gt_batch


class ComputeSegGrads:
    def __init__(self, device):
        self.spatial_gradient = torch_lydorn.kornia.filters.SpatialGradient(mode="scharr", coord="ij", normalized=True, device=device)

    def __call__(self, pred_batch, gt_batch):
        seg = pred_batch["seg"]  # (b, c, h, w)
        seg_grads = 2 * self.spatial_gradient(seg)  # (b, c, 2, h, w), Normalize (kornia normalizes to -0.5, 0.5 for input in [0, 1])
        seg_grad_norm = seg_grads.norm(dim=2)  # (b, c, h, w)
        seg_grads_normed = seg_grads / (seg_grad_norm[:, :, None, ...] + 1e-6)  # (b, c, 2, h, w)
        pred_batch["seg_grads"] = seg_grads
        pred_batch["seg_grad_norm"] = seg_grad_norm
        pred_batch["seg_grads_normed"] = seg_grads_normed
        return pred_batch, gt_batch

class ComputeSegGradsMultiClass:
    def __init__(self, device):
        self.spatial_gradient = torch_lydorn.kornia.filters.SpatialGradient(mode="scharr", coord="ij", normalized=True, device=device)

    def __call__(self, pred_batch, gt_batch):
        for seg_name in ['interior_seg','edge_seg','vertex_seg']:
            seg = pred_batch[seg_name]  # (b, c, h, w)
            if seg.shape[1]>0:
                seg_grads = 2 * self.spatial_gradient(seg)  # (b, c, 2, h, w), Normalize (kornia normalizes to -0.5, 0.5 for input in [0, 1])
                seg_grad_norm = seg_grads.norm(dim=2)  # (b, c, h, w)
                seg_grads_normed = seg_grads / (seg_grad_norm[:, :, None, ...] + 1e-6)  # (b, c, 2, h, w)
                pred_batch[seg_name+"_grads"] = seg_grads
                pred_batch[seg_name+"_grad_norm"] = seg_grad_norm
                pred_batch[seg_name+"_grads_normed"] = seg_grads_normed
        return pred_batch, gt_batch


def build_combined_loss(config):
    pre_processes = []
    loss_funcs = []
    weights = []
    if 'losses_set_name' in config['loss_params']:
        if config['loss_params']['losses_set_name']=='off_nadir':
            partial_compute_seg_loss_weigths = partial(compute_seg_loss_weigths, config=config)
            pre_processes.append(partial_compute_seg_loss_weigths)
            pre_processes.append(compute_gt_field)
            pre_processes.append(ComputeSegGradsMultiClass(config["device"]))
            for loss_name,weight in config['loss_params']['multiloss']['coefs'].items():
                if loss_name=='seg':
                    loss_funcs.append(CategorySegLoss(name='seg'))
                    weights.append(weight)
                if loss_name=='crossfield_align':
                    loss_funcs.append(CrossfieldAlignLossOffNadir(name="crossfield_align"))
                    weights.append(weight)
                if loss_name=='crossfield_align90':
                    loss_funcs.append(CrossfieldAlign90LossOffNadir(name="crossfield_align90"))
                    weights.append(weight)
                if loss_name=='crossfield_smooth':
                    loss_funcs.append(CrossfieldSmoothLoss(name="crossfield_smooth"))
                    weights.append(weight)

                if loss_name=='seg_interior_crossfield':
                    loss_funcs.append(CategorySegCrossfieldLoss(name="seg_interior_crossfield", pred_type='interior',
                                                                pred_channel=[0,1,2],except_edge_mask_channel=3))# 是否可以一次算多个channel？
                    weights.append(config["loss_params"]["multiloss"]["coefs"]["seg_interior_crossfield"])
                if loss_name=='seg_edge_crossfield':
                    loss_funcs.append(CategorySegCrossfieldLoss(name="seg_edge_crossfield",
                                                                pred_type='edge',pred_channel=[0,1,2]))
                    weights.append(config["loss_params"]["multiloss"]["coefs"]["seg_edge_crossfield"])
                if loss_name=='seg_interior_horizontal':
                    loss_funcs.append(HorizontalSegGradLoss(name="seg_interior_crossfield", pred_type='interior',pred_channel=1,edge_mask_channel=3))
                    weights.append(config["loss_params"]["multiloss"]["coefs"]["seg_interior_crossfield"])
                if loss_name=='seg_edge_horizontal':
                    loss_funcs.append(HorizontalSegGradLoss(name="seg_edge_crossfield", pred_type='edge',pred_channel=3))
                    weights.append(config["loss_params"]["multiloss"]["coefs"]["seg_edge_crossfield"])

                if loss_name=='seg_edge_interior':
                    loss_funcs.append(CategorySegEdgeInteriorLoss(name="seg_edge_interior"))
                    weights.append(config["loss_params"]["multiloss"]["coefs"]["seg_edge_interior"])

        # if need_seg_grads:
        #     pre_processes.append(ComputeSegGrads(config["device"]))
        combined_loss = MultiLoss(loss_funcs, weights,
                                  epoch_thresholds=config["loss_params"]["multiloss"]["coefs"]["epoch_thresholds"],
                                  pre_processes=pre_processes)
        return combined_loss

    if config["compute_seg"]:
        partial_compute_seg_loss_weigths = partial(compute_seg_loss_weigths, config=config)
        pre_processes.append(partial_compute_seg_loss_weigths)
        gt_channel_selector = [config["seg_params"]["compute_interior"], config["seg_params"]["compute_edge"], config["seg_params"]["compute_vertex"]]
        loss_funcs.append(SegLoss(name="seg",
                                  gt_channel_selector=gt_channel_selector,
                                  bce_coef=config["loss_params"]["seg_loss_params"]["bce_coef"],
                                  dice_coef=config["loss_params"]["seg_loss_params"]["dice_coef"]))
        weights.append(config["loss_params"]["multiloss"]["coefs"]["seg"])

    if config["compute_crossfield"]:
        pre_processes.append(compute_gt_field)
        loss_funcs.append(CrossfieldAlignLoss(name="crossfield_align"))
        weights.append(config["loss_params"]["multiloss"]["coefs"]["crossfield_align"])
        loss_funcs.append(CrossfieldAlign90Loss(name="crossfield_align90"))
        weights.append(config["loss_params"]["multiloss"]["coefs"]["crossfield_align90"])
        loss_funcs.append(CrossfieldSmoothLoss(name="crossfield_smooth"))
        weights.append(config["loss_params"]["multiloss"]["coefs"]["crossfield_smooth"])

    # --- Coupling losses:
    if config["compute_seg"]:
        need_seg_grads = False
        pred_channel = -1
        # Seg interior <-> Crossfield coupling:
        if config["seg_params"]["compute_interior"] and config["compute_crossfield"]:
            need_seg_grads = True
            pred_channel += 1
            loss_funcs.append(SegCrossfieldLoss(name="seg_interior_crossfield", pred_channel=pred_channel))
            weights.append(config["loss_params"]["multiloss"]["coefs"]["seg_interior_crossfield"])
        # Seg edge <-> Crossfield coupling:
        if config["seg_params"]["compute_edge"] and config["compute_crossfield"]:
            need_seg_grads = True
            pred_channel += 1
            loss_funcs.append(SegCrossfieldLoss(name="seg_edge_crossfield", pred_channel=pred_channel))
            weights.append(config["loss_params"]["multiloss"]["coefs"]["seg_edge_crossfield"])

        # Seg edge <-> seg interior coupling:
        if config["seg_params"]["compute_interior"] and config["seg_params"]["compute_edge"]:
            need_seg_grads = True
            loss_funcs.append(CategorySegEdgeInteriorLoss(name="seg_edge_interior"))
            weights.append(config["loss_params"]["multiloss"]["coefs"]["seg_edge_interior"])

        if need_seg_grads:
            pre_processes.append(ComputeSegGrads(config["device"]))

    combined_loss = MultiLoss(loss_funcs, weights, epoch_thresholds=config["loss_params"]["multiloss"]["coefs"]["epoch_thresholds"], pre_processes=pre_processes)
    return combined_loss


# --- Specific losses --- #
class SegLoss(Loss):
    def __init__(self, name, gt_channel_selector, bce_coef=0.5, dice_coef=0.5):
        """
        :param name:
        :param gt_channel_selector: used to select which channels gt_polygons_image to use to compare to predicted seg
                                    (see docstring of method compute() for more details).
        """
        super(SegLoss, self).__init__(name)
        self.gt_channel_selector = gt_channel_selector
        self.bce_coef = bce_coef
        self.dice_coef = dice_coef

    def compute(self, pred_batch, gt_batch):
        """
        seg and gt_polygons_image do not necessarily have the same channel count.
        gt_selector is used to select which channels of gt_polygons_image to use.
        For example, if seg has C_pred=2 (interior and edge) and
        gt_polygons_image has C_gt=3 (interior, edge and vertex), use gt_channel_selector=slice(0, 2)

        @param pred_batch: key "seg" is shape (N, C_pred, H, W)
        @param gt_batch: key "gt_polygons_image" is shape (N, C_gt, H, W)
        @return:
        """
        # print(self.name)
        pred_seg = pred_batch["seg"]
        gt_seg = gt_batch["gt_polygons_image"][:, self.gt_channel_selector, ...]
        weights = gt_batch["seg_loss_weights"][:, self.gt_channel_selector, ...]
        dice = measures.dice_loss(pred_seg, gt_seg)
        mean_dice = torch.mean(dice)
        mean_cross_entropy = F.binary_cross_entropy(pred_seg, gt_seg, weight=weights, reduction="mean")

        # Display:
        # dispaly_pred_seg = pred_seg[0, 0].cpu().detach().numpy()
        # print(f'{self.name}_pred:', dispaly_pred_seg.shape, dispaly_pred_seg.min(), dispaly_pred_seg.max())
        # skimage.io.imsave(f'{self.name}_pred.png', dispaly_pred_seg)
        # dispaly_gt_seg = gt_seg[0].cpu().detach().numpy()
        # skimage.io.imsave(f'{self.name}_gt.png', dispaly_gt_seg)

        return self.bce_coef * mean_cross_entropy + self.dice_coef * mean_dice

class CategorySegLoss(Loss):
    def __init__(self, name, gt_channel_selector=[True,True,False], bce_coef=0.5, dice_coef=0.5):
        """
        于seg loss的区别在于这里使用的是每个channel有多个类别，gt用数值代表了不同类别
        :param name:
        :param gt_channel_selector: used to select which channels gt_polygons_image to use to compare to predicted seg
                                    (see docstring of method compute() for more details).
        """
        super(CategorySegLoss, self).__init__(name)
        self.gt_channel_selector = gt_channel_selector# interior,edge,vertex channels
        self.bce_coef = bce_coef
        self.dice_coef = dice_coef

    def compute(self, pred_batch, gt_batch):
        """
        seg and gt_polygons_image do not necessarily have the same channel count.
        gt_selector is used to select which channels of gt_polygons_image to use.
        For example, if seg has C_pred=2 (interior and edge) and
        gt_polygons_image has C_gt=3 (interior, edge and vertex), use gt_channel_selector=slice(0, 2)

        @param pred_batch: key "seg" is shape (N, C_pred, H, W)
        @param gt_batch: key "gt_polygons_image" is shape (N, C_gt, H, W)
        @return:
        """
        # print(self.name)
        mean_cross_entropy=0
        mean_dice=0
        if self.gt_channel_selector[0]:# interior
            # pred_seg = pred_batch["interior_seg"]
            pred_seg = pred_batch["interior_seg"] # test
            gt_seg = gt_batch["gt_polygons_image"][:, 0, ...].long()# CudaDataAugmentation中会将其转换为float用于双线性插值
            weights = gt_batch["seg_loss_weights"]['interior']
            # mean_cross_entropy += F.cross_entropy(pred_seg, gt_seg, weight=weights, reduction="mean")
            mean_cross_entropy += F.cross_entropy(pred_seg, gt_seg,  weights,reduction="mean")
            gt_onehot=F.one_hot(gt_seg,num_classes=pred_seg.shape[1]).permute(0,3,1,2)
            dice=measures.dice_loss(pred_seg[:,1:],gt_onehot[:,1:])# 计算background之外的有意义的类的dice
            mean_dice += torch.mean(dice)
            # todo:dice loss of multi class
        if self.gt_channel_selector[1]:
            pred_seg = pred_batch["edge_seg"]
            gt_seg = gt_batch["gt_polygons_image"][:, 1, ...].long()
            # weights = gt_batch["seg_loss_weights"]['edge']
            # mean_cross_entropy += F.cross_entropy(pred_seg, gt_seg, weight=weights, reduction="mean")
            gt_onehot=F.one_hot(gt_seg,num_classes=pred_seg.shape[1]).permute(0,3,1,2)
            mean_cross_entropy += F.binary_cross_entropy(pred_seg[:, 1], gt_onehot.float()[:, 1], reduction="mean")
            mean_cross_entropy += F.binary_cross_entropy(pred_seg[:, 2], gt_onehot.float()[:, 2], reduction="mean")
            mean_cross_entropy += F.binary_cross_entropy(pred_seg[:, 3], gt_onehot.float()[:, 3], reduction="mean")
            dice=measures.dice_loss(pred_seg[:,1:],gt_onehot[:,1:])# 计算background之外的有意义的类的dice
            mean_dice += torch.mean(dice)
        if self.gt_channel_selector[2]:
            pred_seg = pred_batch["vertex_seg"]
            gt_seg = gt_batch["gt_polygons_image"][:, 2, ...].long()
            weights = gt_batch["seg_loss_weights"]['vertex']
            mean_cross_entropy += F.cross_entropy(pred_seg, gt_seg, weight=weights, reduction="mean")
            gt_onehot=F.one_hot(gt_seg,num_classes=pred_seg.shape[1]).permute(0,3,1,2)
            dice=measures.dice_loss(pred_seg[:,1:],gt_onehot[:,1:])# 计算background之外的有意义的类的dice
            mean_dice += torch.mean(dice)


        # Display:
        # dispaly_pred_seg = pred_seg[0, 0].cpu().detach().numpy()
        # print(f'{self.name}_pred:', dispaly_pred_seg.shape, dispaly_pred_seg.min(), dispaly_pred_seg.max())
        # skimage.io.imsave(f'{self.name}_pred.png', dispaly_pred_seg)
        # dispaly_gt_seg = gt_seg[0].cpu().detach().numpy()
        # skimage.io.imsave(f'{self.name}_gt.png', dispaly_gt_seg)

        return self.bce_coef * mean_cross_entropy + self.dice_coef * mean_dice

class CrossfieldAlignLoss(Loss):
    def __init__(self, name):
        super(CrossfieldAlignLoss, self).__init__(name)

    def compute(self, pred_batch, gt_batch):
        c0 = pred_batch["crossfield"][:, :2]
        c2 = pred_batch["crossfield"][:, 2:]
        z = gt_batch["gt_field"]
        gt_polygons_image = gt_batch["gt_polygons_image"]
        assert 2 <= gt_polygons_image.shape[1], \
            "gt_polygons_image should have at least 2 channels for interior and edges"
        gt_edges = gt_polygons_image[:, 1, ...]
        align_loss = frame_field_utils.framefield_align_error(c0, c2, z, complex_dim=1)
        avg_align_loss = torch.mean(align_loss * gt_edges)

        self.extra_info["gt_field"] = gt_batch["gt_field"]
        return avg_align_loss


class CrossfieldAlign90Loss(Loss):
    def __init__(self, name):
        super(CrossfieldAlign90Loss, self).__init__(name)

    def compute(self, pred_batch, gt_batch):
        c0 = pred_batch["crossfield"][:, :2]
        c2 = pred_batch["crossfield"][:, 2:]
        z = gt_batch["gt_field"]
        z_90deg = torch.cat((- z[:, 1:2, ...], z[:, 0:1, ...]), dim=1)
        gt_polygons_image = gt_batch["gt_polygons_image"]
        assert gt_polygons_image.shape[1] == 3, \
            "gt_polygons_image should have 3 channels for interior, edges and vertices"
        gt_edges = gt_polygons_image[:, 1, ...]
        gt_vertices = gt_polygons_image[:, 2, ...]
        gt_edges_minus_vertices = gt_edges - gt_vertices
        gt_edges_minus_vertices = gt_edges_minus_vertices.clamp(0, 1)
        align90_loss = frame_field_utils.framefield_align_error(c0, c2, z_90deg, complex_dim=1)
        avg_align90_loss = torch.mean(align90_loss * gt_edges_minus_vertices)
        return avg_align90_loss

class CrossfieldAlignLossOffNadir(Loss):
    def __init__(self, name):
        super(CrossfieldAlignLossOffNadir, self).__init__(name)

    def compute(self, pred_batch, gt_batch):
        c0 = pred_batch["crossfield"][:, :2]
        c2 = pred_batch["crossfield"][:, 2:]
        z = gt_batch["gt_field"]
        gt_polygons_image = gt_batch["gt_polygons_image"]
        assert 2 <= gt_polygons_image.shape[1], \
            "gt_polygons_image should have at least 2 channels for interior and edges"
        align_loss = frame_field_utils.framefield_align_error(c0, c2, z, complex_dim=1)
        mask=torch.logical_or(torch.logical_or(gt_polygons_image[:, 1, ...]==1,gt_polygons_image[:, 1, ...]==2),
                              gt_polygons_image[:, 0, ...]==1)
        avg_align_loss = torch.mean(align_loss * mask)

        self.extra_info["gt_field"] = gt_batch["gt_field"]
        return avg_align_loss


class CrossfieldAlign90LossOffNadir(Loss):
    def __init__(self, name):
        super(CrossfieldAlign90LossOffNadir, self).__init__(name)

    def compute(self, pred_batch, gt_batch):
        c0 = pred_batch["crossfield"][:, :2]
        c2 = pred_batch["crossfield"][:, 2:]
        z = gt_batch["gt_field"]
        z_90deg = torch.cat((- z[:, 1:2, ...], z[:, 0:1, ...]), dim=1)
        gt_polygons_image = gt_batch["gt_polygons_image"]
        assert gt_polygons_image.shape[1] == 3, \
            "gt_polygons_image should have 3 channels for interior, edges and vertices"
        gt_vertices = gt_polygons_image[:, 2, ...]
        mask=torch.logical_or(torch.logical_or(gt_polygons_image[:, 1, ...]==1,gt_polygons_image[:, 1, ...]==2),
                              gt_polygons_image[:, 0, ...]==1)
        mask = mask.to(float) - gt_vertices
        mask = mask.clamp(0, 1)
        align90_loss = frame_field_utils.framefield_align_error(c0, c2, z_90deg, complex_dim=1)
        avg_align90_loss = torch.mean(align90_loss * mask)
        return avg_align90_loss

class CrossfieldSmoothLoss(Loss):
    def __init__(self, name):
        super(CrossfieldSmoothLoss, self).__init__(name)
        self.laplacian_penalty = frame_field_utils.LaplacianPenalty(channels=4)

    def compute(self, pred_batch, gt_batch):
        c0c2 = pred_batch["crossfield"]
        gt_polygons_image = gt_batch["gt_polygons_image"]
        gt_edges_inv = 1 - gt_polygons_image[:, 1, ...]
        penalty = self.laplacian_penalty(c0c2)
        # avg_penalty = torch.mean(penalty * gt_edges_inv[:, None, ...])
        avg_penalty = torch.mean(penalty)# no mask
        return avg_penalty


class SegCrossfieldLoss(Loss):
    def __init__(self, name, pred_channel):
        super(SegCrossfieldLoss, self).__init__(name)
        self.pred_channel = pred_channel

    def compute(self, pred_batch, gt_batch):
        # TODO: don't apply on corners: corner_map = gt_batch["gt_polygons_image"][:, 2, :, :]
        # TODO: apply on all seg at once? Like seg is now?
        c0 = pred_batch["crossfield"][:, :2]
        c2 = pred_batch["crossfield"][:, 2:]
        seg_slice_grads_normed = pred_batch["seg_grads_normed"][:, self.pred_channel, ...]
        seg_slice_grad_norm = pred_batch["seg_grad_norm"][:, self.pred_channel, ...]
        align_loss = frame_field_utils.framefield_align_error(c0, c2, seg_slice_grads_normed, complex_dim=1)
        # normed_align_loss = align_loss * seg_slice_grad_norm
        # avg_align_loss = torch.sum(normed_align_loss) / (torch.sum(seg_slice_grad_norm) + 1e-6)
        avg_align_loss = torch.mean(align_loss * seg_slice_grad_norm.detach())
        # (prev line) Don't back-propagate to seg_slice_grad_norm so that seg smoothness is not encouraged

        # Save extra info for viz:
        self.extra_info["seg_slice_grads"] = pred_batch["seg_grads"][:, self.pred_channel, ...]
        return avg_align_loss

class CategorySegCrossfieldLoss(Loss):
    def __init__(self, name, pred_type,pred_channel,except_edge_mask_channel=None):
        super(CategorySegCrossfieldLoss, self).__init__(name)
        self.pred_type = pred_type
        self.except_edge_mask_channel = except_edge_mask_channel
        if isinstance(pred_channel,list):
            self.pred_channels = pred_channel
        else:
            self.pred_channels = [pred_channel]

    def compute(self, pred_batch, gt_batch):
        # TODO: don't apply on corners: corner_map = gt_batch["gt_polygons_image"][:, 2, :, :]
        # TODO: apply on all seg at once? Like seg is now?
        c0 = pred_batch["crossfield"][:, :2]
        c2 = pred_batch["crossfield"][:, 2:]
        align_loss=0
        for pred_channel in self.pred_channels:
            seg_slice_grads_normed = pred_batch[self.pred_type+"_seg_grads_normed"][:, pred_channel, ...]
            seg_slice_grad_norm = pred_batch[self.pred_type+"_seg_grad_norm"][:, pred_channel, ...]
            align_loss += frame_field_utils.framefield_align_error(c0, c2, seg_slice_grads_normed, complex_dim=1)* seg_slice_grad_norm.detach()
        if self.except_edge_mask_channel is not None:
            mask=1-pred_batch['edge_seg'][:, self.except_edge_mask_channel, ...]
            # normed_align_loss = align_loss * seg_slice_grad_norm
            # avg_align_loss = torch.sum(normed_align_loss) / (torch.sum(seg_slice_grad_norm) + 1e-6)
            avg_align_loss = torch.mean(align_loss*mask)
        else:
            avg_align_loss = torch.mean(align_loss)
        return avg_align_loss

class HorizontalSegGradLoss(Loss):
    def __init__(self, name, pred_type,pred_channel,edge_mask_channel=None):
        super(HorizontalSegGradLoss, self).__init__(name)
        self.pred_type=pred_type
        self.edge_mask_channel = edge_mask_channel
        self.pred_channel = pred_channel

    def compute(self, pred_batch, gt_batch):
        seg_slice_grads = pred_batch[self.pred_type+"_seg_grads_normed"][:, self.pred_channel, ...]
        if self.edge_mask_channel is not None:
            mask=pred_batch['edge_seg'][:,self.edge_mask_channel]
            avg_align_loss = torch.mean(torch.abs(seg_slice_grads[:,1])*mask)
        else:
            avg_align_loss = torch.mean(torch.abs(seg_slice_grads[:,1]))
        return avg_align_loss



class SegEdgeInteriorLoss(Loss):
    """
    Enforce seg edge to be equal to interior grad norm except inside buildings
    """

    def __init__(self, name):
        super(SegEdgeInteriorLoss, self).__init__(name)

    def compute(self, pred_batch, batch):
        seg_interior = pred_batch["seg"][:, 0, ...]
        seg_edge = pred_batch["seg"][:, 1, ...]
        seg_interior_grad_norm = pred_batch["seg_grad_norm"][:, 0, ...]
        raw_loss = torch.abs(seg_edge - seg_interior_grad_norm)
        # Apply the loss only on interior boundaries and outside of objects
        outside_mask = (torch.cos(np.pi * seg_interior) + 1) / 2
        boundary_mask = (1 - torch.cos(np.pi * seg_interior_grad_norm)) / 2
        mask = torch.max(outside_mask, boundary_mask).float()
        avg_loss = torch.mean(raw_loss * mask)
        return avg_loss


class CategorySegEdgeInteriorLoss(Loss):
    """
    Enforce seg edge to be equal to interior grad norm except inside buildings
    """

    def __init__(self, name):
        super(CategorySegEdgeInteriorLoss, self).__init__(name)

    def compute(self, pred_batch, batch):
        # 暂时将参数写死了
        # seg_interior_c1 = pred_batch["interior_seg"][:, 1, ...]
        # seg_interior_c2 = pred_batch["interior_seg"][:, 2, ...]
        seg_edge_x_between = pred_batch["edge_seg"][:, 1, ...]
        seg_edge_x_roof_right = pred_batch["edge_seg"][:, 2, ...]
        seg_edge_y_facade_up_down = pred_batch["edge_seg"][:, 3, ...]
        seg_interior_roof_x_grad = pred_batch["interior_seg_grads"][:, 2, 1, ...]# dim1:1-facade 2:roof dim 2:1-x 0-y
        seg_interior_facade_x_grad = pred_batch["interior_seg_grads"][:, 1, 1, ...]
        seg_interior_facade_y_grad = pred_batch["interior_seg_grads"][:, 1, 0, ...]
        seg_interior_facade_y_grad_normed = pred_batch["interior_seg_grads_normed"][:, 1, 0, ...]
        raw_loss_1 = torch.abs(seg_edge_x_roof_right - F.relu(-seg_interior_roof_x_grad))
        mask_1= (1 - torch.cos(np.pi * F.relu(-seg_interior_roof_x_grad))) / 2
        raw_loss_2 = torch.abs(seg_edge_x_between - F.relu(seg_interior_roof_x_grad))
        mask_2= (1 - torch.cos(np.pi * F.relu(seg_interior_roof_x_grad))) / 2
        raw_loss_3 = torch.abs(seg_edge_x_between - F.relu(-seg_interior_facade_x_grad))
        mask_3= (1 - torch.cos(np.pi * F.relu(-seg_interior_facade_x_grad))) / 2
        raw_loss_4 = torch.abs(F.relu(seg_interior_roof_x_grad) - F.relu(-seg_interior_facade_x_grad))
        # raw_loss_5 = torch.abs(seg_edge_y_facade_up_down - torch.abs(seg_interior_facade_y_grad))
        facade_horzontal_edge=(1 - torch.cos(10*np.pi * (F.relu(torch.abs(seg_interior_facade_y_grad_normed)-0.9)))) / 2
        raw_loss_5 = torch.abs(seg_edge_y_facade_up_down - facade_horzontal_edge)
        # mask_5= (1 - torch.cos(np.pi * torch.abs(seg_interior_facade_y_grad))) / 2
        # Apply the loss only on interior boundaries and outside of objects
        # total_loss=raw_loss_1*mask_1+raw_loss_2*mask_2+raw_loss_3*mask_3+raw_loss_4+raw_loss_5*mask_5
        total_loss=raw_loss_1+raw_loss_2+raw_loss_3+raw_loss_4+raw_loss_5
        avg_loss = torch.mean(total_loss)
        return avg_loss


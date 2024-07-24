# encoding:utf-8
import math

import mmcv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
import cv2

from ...core import tensor2imgs


@DETECTORS.register_module()
class SingleStageRegressor(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 regress_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageRegressor, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self.regress_head = build_head(regress_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(SingleStageRegressor, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.regress_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x


    def forward_train(self,
                      img,
                      img_metas=None,
                      **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_valuees (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_valuees_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # self.show_label(img,img_metas,**kwargs)
        x = self.extract_feat(img)
        losses = dict()
        regress_losses = self.regress_head.forward_train(x, **kwargs)
        losses.update(regress_losses)
        return losses

    def show_label(self,img,img_metas,**kwargs):
        self.show_result(tensor2imgs(img[:1], **img_metas[0]['img_norm_cfg'])[0],{'angle':kwargs['gt_angle'][0].item()} )

    def simple_test(self, img, img_metas, rescale=False,**kwargs):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
            **kwargs:数据中给出其他信息时，用这个接收防止报错
        Returns:
            dict{'angle':outs} 输出形式与数据集bonai的evaluate配合
        """
        x = self.extract_feat(img)
        outs,cos_sin_out = self.regress_head(x)
        result=[]
        # 这里与loft_head略有不同， 因为loft_head是对单个图片处理的，但是这里是多个图片，但是还是要保存到list（dict）类型的result中去以保证后续的操作形式一致（show_result等）
        for single_out in outs:# 从batch处理中拆分出各个图片的结果，分别放到list（dict）类型的result中去
            result.append({'angle':single_out.item(),'xy_dir':cos_sin_out})
        return result

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        raise NotImplementedError

    def show_result(self,
                    img,
                    result,**kwargs):
        """
        Args:
            img: ndarray[H,W,3],输入图像，已经转换成HWC的图像形式并将norm恢复了（tensor2imgs方法）
            result: dict{'angle':tesnor or float} 角度预测结果
        """
        img = mmcv.imread(img)[:,:,::-1]# bgr->rgb
        plt.imshow(img)
        angle_result = result.get('angle', None)
        if angle_result is not None:
            if isinstance(angle_result,torch.Tensor):
                angle_result=angle_result.cpu().numpy()
            # angle_result = -90 / 180 * math.pi
            img_shape=img.shape[:2]
            arrow_interval=100
            arrow_len=7
            # ref:https://juejin.cn/post/7034117434659471397
            x, y = np.meshgrid(np.arange(0, img_shape[0], arrow_interval),np.arange(0, img_shape[1], arrow_interval))
            u, v = (arrow_len*np.cos(angle_result),-arrow_len*np.sin(angle_result))# 由于绘制图像时还是图像上方y大，而quiver场显示是y的下方大，因此要对y方向的offset反向
            plt.quiver(x, y, u, v, color="pink", pivot="tail", units="inches")
            plt.scatter(x, y, color="b", s=0.05)
        plt.show()
        angle_result=angle_result*180/np.pi
        rotation_M=cv2.getRotationMatrix2D((0,0),float(angle_result),scale=1)# cv2的旋转角度是角度制，要从弧度制转换过来，旋转角度是逆时针
        rotation_M=np.vstack([rotation_M,np.array([0,0,1])])
        corner_pts=np.float32([[0,0],[img.shape[0],0],[0,img.shape[1]],[img.shape[0],img.shape[1]]])
        corner_pts=np.float32(corner_pts).reshape(-1,1,2)
        rot_corner_pts=cv2.perspectiveTransform(corner_pts,rotation_M)
        left=rot_corner_pts[:,:,0].min()
        top=rot_corner_pts[:,:,1].min()
        width=rot_corner_pts[:,:,0].max()-left
        height=rot_corner_pts[:,:,1].max()-top
        rotation_M[0,2]=-left
        rotation_M[1,2]=-top
        rot_img=cv2.warpPerspective(img, rotation_M, (math.ceil(width), math.ceil(height)))
        plt.imshow(rot_img)
        plt.show()



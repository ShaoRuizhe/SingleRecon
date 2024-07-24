
import torch
from .unet_vflow import UnetVFLOW
from .base import BaseDetector
from ..builder import DETECTORS

@DETECTORS.register_module()
class UnetVFLOWAngle(UnetVFLOW,BaseDetector):

    def forward(self, img, img_metas=None, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        return super(UnetVFLOWAngle, self).forward(img,img_metas,return_loss,**kwargs)

    def forward_train(self, x,img_metas,gt_angle,*args,**kwargs):
        features = self.encoder(x)
        xydir = self.xydir_head(features[-1])
        gt_angle = torch.tensor(gt_angle).cuda()
        cos_sin_gt = torch.vstack((torch.cos(gt_angle), torch.sin(gt_angle))).permute(1, 0)
        loss=torch.nn.MSELoss()(xydir,cos_sin_gt)
        return {'angle_regress_loss':loss}

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
        features = self.encoder(img)
        xydir = self.xydir_head(features[-1])
        cos_x = xydir[:, 0]
        sin_x = xydir[:, 1]
        outs = torch.angle(cos_x + sin_x * 1j)
        result=[]
        # 这里与loft_head略有不同， 因为loft_head是对单个图片处理的，但是这里是多个图片，但是还是要保存到list（dict）类型的result中去以保证后续的操作形式一致（show_result等）
        for i,single_out in enumerate(outs):# 从batch处理中拆分出各个图片的结果，分别放到list（dict）类型的result中去
            result.append({'angle':single_out.item(),'xy_dir':xydir[i]})
        return result

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        raise NotImplementedError

    def extract_feat(self, img):
        raise NotImplementedError

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        pass
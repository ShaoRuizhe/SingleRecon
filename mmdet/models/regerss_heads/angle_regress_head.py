# encoding:utf-8
import torch.linalg
from torch import nn
from torch.nn import Conv2d,AvgPool2d

from mmdet.models import build_loss
from mmdet.models.backbones.resnet import BasicBlock,Bottleneck
from mmdet.models.utils import ResLayer
from ..builder import HEADS

@HEADS.register_module()
class AngleRegressHead(nn.Module):
    def __init__(self,in_channels,loss_angle=dict(type='MSELoss')):
        super(AngleRegressHead, self).__init__()
        # todo:减小num_blocks甚至只使用一个BasicBlock
        self.res_layer=ResLayer(block=BasicBlock,inplanes=in_channels,planes=16,num_blocks=3,stride=2)# 2048*32*32->64*16*16
        # self.res_layer=ResLayer(block=Bottleneck,inplanes=in_channels,planes=4,num_blocks=3,stride=2)# 2048*32*32->64*16*16 #注意Bottleneck的输出channel是4*planes
        self.conv_to_1=Conv2d(in_channels=16,out_channels=2,kernel_size=3,stride=1,padding=1)# 64*16*16->2*16*16
        self.pool=AvgPool2d(kernel_size=16)# 2*16*16->2
        self.loss=build_loss(loss_angle)

    def init_weights(self, pretrained=None):
        pass

    def forward_train(self,x,gt_angle):
        '''
        用cos sin计算loss
        Args:
            x: tensor[batch_size]
            gt_angle :list[tensor]

        Returns:

        '''
        result=self.forward_cos_sin(x)
        gt_angle=torch.tensor(gt_angle).cuda()
        cos_sin_gt=torch.vstack((torch.cos(gt_angle),torch.sin(gt_angle))).permute(1,0)
        loss=self.loss(result,cos_sin_gt)
        return {'angle_regress_loss':loss}

    def forward_cos_sin(self,x):
        x = self.res_layer(x[0])
        x = self.conv_to_1(x)
        # x_norm = torch.linalg.vector_norm(x, dim=1)
        # x = x / x_norm
        out = self.pool(x)
        out = out.squeeze(2).squeeze(2)
        return out

    def forward(self,x):# 其实就是forward_test
        cos_sin_out=self.forward_cos_sin(x)
        cos_x=cos_sin_out[:,0]
        sin_x=cos_sin_out[:,1]
        out=torch.angle(cos_x+sin_x*1j)# 构造一个虚数，用于获取角度
        # return out
        return out,cos_sin_out
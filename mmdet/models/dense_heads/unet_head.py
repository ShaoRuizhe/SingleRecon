#encoding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import HEADS, build_loss
from torchvision.transforms import Resize,InterpolationMode

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv
    input:x1:[b,in_channel,w,h] x2:[b,in_channel/2,2w,2h] output:[b,out_channel,2w,2h]
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


@HEADS.register_module()
class UNetHead(nn.Module):
    def __init__(self, n_channels=[2048,1024,512,256], n_classes=4, bilinear=False,# [2048,1024,512,256] is output of resnet50
                 loss_crossfield_align=dict(type='CrossfieldAlignOffNadirLoss'),loss_crossfield_smooth=None):
        super(UNetHead, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1 # 使用反卷积时，用反卷积进行输入降维，使用双线性时，就需要输出的时候提前为下一层降维
        self.up1 = (Up(self.n_channels[0], self.n_channels[1] // factor, bilinear))
        self.up2 = (Up(self.n_channels[1], self.n_channels[2] // factor, bilinear))
        self.up3 = (Up(self.n_channels[2], self.n_channels[3] // factor, bilinear))
        # self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(self.n_channels[3], n_classes))

        self.loss_crossfield_align = build_loss(loss_crossfield_align)
        self.resize=Resize(size=(256, 256), interpolation=InterpolationMode.NEAREST)
        self.loss_crossfield_smooth=build_loss(loss_crossfield_smooth) if loss_crossfield_smooth is not None else None

    def forward(self, x):
        x1, x2, x3, x4=x
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

    def forward_train(self, x,gt_crossfields,gt_segs):
        pre_crossfield = self(x)
        if not gt_crossfields.shape[-2:] == gt_segs.shape[1:3]:
            gt_segs = self.resize(gt_segs.permute(0,3,1,2)).permute(0,2,3,1)
        align_loss=self.loss_crossfield_align(pre_crossfield, gt_crossfields, gt_segs) # mask
        losses_dict=dict(loss_crossfield_align=align_loss)
        if self.loss_crossfield_smooth is not None:
            loss_crossfield_smooth=self.loss_crossfield_smooth(pre_crossfield)
            losses_dict.update(dict(loss_crossfield_smooth=loss_crossfield_smooth))
        return losses_dict,pre_crossfield


@HEADS.register_module()
class UNetHead1024(nn.Module):
    def __init__(self, n_channels=[2048,1024,512,256,8], n_classes=4, bilinear=False,# [2048,1024,512,256] is output of resnet50
                 loss_crossfield_align=dict(type='CrossfieldAlignOffNadirLoss'),loss_crossfield_smooth=None):
        super(UNetHead1024, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1 # 使用反卷积时，用反卷积进行输入降维，使用双线性时，就需要输出的时候提前为下一层降维
        self.up1 = (Up(self.n_channels[0], self.n_channels[1] // factor, bilinear))
        self.up2 = (Up(self.n_channels[1], self.n_channels[2] // factor, bilinear))
        self.up3 = (Up(self.n_channels[2], self.n_channels[3] // factor, bilinear))
        self.upsample=nn.ConvTranspose2d(self.n_channels[3], self.n_channels[4], kernel_size=4, stride=4)
        # self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(self.n_channels[4], n_classes))

        self.loss_crossfield_align = build_loss(loss_crossfield_align)
        self.loss_crossfield_smooth=build_loss(loss_crossfield_smooth) if loss_crossfield_smooth is not None else None

    def forward(self, x):
        x1, x2, x3, x4=x
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x=self.upsample(x)
        logits = self.outc(x)
        return logits

    def forward_train(self, x,gt_crossfields,gt_segs):
        pre_crossfield = self(x)
        align_loss=self.loss_crossfield_align(pre_crossfield, gt_crossfields, gt_segs) # mask
        losses_dict=dict(loss_crossfield_align=align_loss)
        if self.loss_crossfield_smooth is not None:
            loss_crossfield_smooth=self.loss_crossfield_smooth(pre_crossfield)
            losses_dict.update(dict(loss_crossfield_smooth=loss_crossfield_smooth))
        return losses_dict,pre_crossfield
# Copyright: njfh-AIlab-group-3
import math

import torch
import torch.nn as nn

import logging
from ..builder import BACKBONES
from mtcv.cnn import ConvModule, build_conv_layer, build_norm_layer, build_act_layer
from mtcv.runner import load_checkpoint
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,  # in channels
                 planes,  # out channels
                 stride=1,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type="BN", momentum=0.1),
                 act_cfg=dict(type='ReLU')):
        super(BasicBlock, self).__init__()
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(conv_cfg, inplanes, planes, 3,  # 3*3 conv
                                      stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.add_module(self.norm1_name, norm1)

        self.conv2 = build_conv_layer(  # 3*3 conv
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.act = build_act_layer(act_cfg)
        self.stride = stride
        self.dilation = dilation

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += residual
        out = self.act(out)

        return out


class Root(nn.Module):
    """
    concat nodes, and perform conv op.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 residual,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.1),
                 act_cfg=dict(type='ReLU')):
        super(Root, self).__init__()
        self.conv = build_conv_layer(conv_cfg, in_channels, out_channels, 1, stride=1, bias=False,
                                     padding=(kernel_size - 1) // 2)
        self.bn = build_norm_layer(norm_cfg, out_channels)
        self.act = build_act_layer(act_cfg)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.act(x)
        return x


class Tree(nn.Module):
    """
    The tree structure of DLA
    """

    def __init__(self,
                 levels,
                 block,
                 in_channels,
                 out_channels,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 dilation=1,
                 root_residual=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.1),
                 act_cfg=dict(type='ReLU')):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual, conv_cfg, norm_cfg, act_cfg)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                build_norm_layer(norm_cfg, out_channels))

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    """
    Deep Layer Aggregation network.
    This is  a DLA-34 implementation.

    output stride: [2, 4, 8, 16, 32]
    """

    def __init__(self,
                 levels=[1, 1, 1, 2, 2, 1],
                 channels=[16, 32, 64, 128, 256, 512],
                 block=BasicBlock,
                 norm_cfg=dict(type='BN', requires_grad=True, momentum=0.1),
                 residual_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=0.1),
            nn.ReLU(inplace=True))

        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])

        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)

        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root,norm_cfg=norm_cfg)

        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root,norm_cfg=norm_cfg)

        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root,norm_cfg=norm_cfg)

        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root,norm_cfg=norm_cfg)

    def init_weights(self, pretrained=None):
        if pretrained:
            if isinstance(pretrained, str):
                logger = logging.getLogger()
                load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.append(ConvModule(inplanes,
                                      planes,
                                      kernel_size=3,
                                      stride=stride if i == 0 else 1,
                                      padding=dilation,
                                      bias=False,
                                      dilation=dilation,
                                      norm_cfg=dict(type='BN', momentum=0.1)))
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, f'level{i}')(x)
            y.append(x)
        return y


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=0.1),
            nn.ReLU(inplace=True))
        self.conv = build_conv_layer(dict(type='DCNv2'), chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1,
                                     deform_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


def fill_up_weights(up):
    """
    bilinear interpolation initialization.
    """
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)

            fill_up_weights(up)

            setattr(self, f'proj_{i}', proj)
            setattr(self, f'up_{i}', up)
            setattr(self, f'node_{i}', node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, f'up_{i - startp}')
            project = getattr(self, f'proj_{i - startp}')
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, f'node_{i - startp}')
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp  # 2
        if in_channels is None:
            in_channels = channels  # [64, 128, 256, 512]
        self.channels = channels
        channels = list(channels)
        # [1, 2, 4, 8]
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):  # range(4-1)
            j = -i - 2  # [-2, -3, -4]
            setattr(self, f'ida_{i}',
                    IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j]))  # scale[1, 2, 4] ,[1 ,2], [1]
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):  # range(6 -2 -1) 0 1 2
            ida = getattr(self, f'ida_{i}')
            # layers: [level0:level6], 6-i-2,6)
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


@BACKBONES.register_module('DLANet')
class DLASeg(nn.Module):
    """
    The Segmentation network of DLA-34, which used for CenterNet object detection.
    Some modification has made to better extract the features.
    The conv are replaced with 3 x 3 deformable conv at every upsampling layer.
    one 3 x 3 convo with 256 channel are added before each output head.

    """

    def __init__(self,
                 down_ratio,
                 last_level,
                 levels=[1, 1, 1, 2, 2, 1],
                 channels=[16, 32, 64, 128, 256, 512],
                 norm_cfg=dict(type='BN', requires_grad=True, momentum=0.1),
                 out_channel=0):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))  # 2
        self.last_level = last_level  # 5
        self.base = DLA(levels=levels, channels=channels, norm_cfg=norm_cfg)
        # [16, 32, 64, 128, 256, 512]
        channels = self.base.channels

        # [1, 2, 4, 8]
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]

        # DLAUp(2,[64,128,256,512], [1,2,4,8]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            # out_channel = 64
            out_channel = channels[self.first_level]

        # IDAUp(64, [64,128,256], [0, 1, 2]
        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            self.base.init_weights(pretrained)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return y

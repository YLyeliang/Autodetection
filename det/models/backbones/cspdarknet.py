import logging

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mtcv.cnn import build_conv_layer, build_norm_layer, build_act_layer, ConvModule
from mtcv.cnn.weight_init import kaiming_init, constant_init

from mtcv.runner import load_checkpoint

from ..builder import BACKBONES


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,  # in channels
                 planes,  # out channels
                 stride=1,
                 dilation=1,
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=dict(type="BN"),
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
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.act(out)

        return out


def make_res_layer(block,
                   planes,
                   blocks,
                   dilation=1,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   act_cfg=dict(type='ReLU')):
    downsample = None
    # check whether downsample,at first iteration,make sure the input of next iteration have same channels with outputs.
    # e.g if input is 64, during bottlenck iteration,the first block
    # 64(x) -1*1*planes-3*3*planes - 1*1*(planes*expansion)-out down_x=downsample(64(x) -planes*expansion) summation(out,down_x).

    layers = []
    inplanes = planes * block.expansion

    for i in range(1, blocks):
        layers.append(block(inplanes, planes, 1, dilation, conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg, act_cfg=act_cfg))

    return nn.Sequential(*layers)


class StemLayer(nn.Module):

    def __init__(self,
                 planes,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Mish')):
        super(StemLayer, self).__init__()
        self.conv1 = ConvModule(3, planes, kernel_size=3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        self.conv2 = ConvModule(planes, planes * 2, kernel_size=3, stride=2, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.part_one = ConvModule(planes * 2, planes * 2, kernel_size=1, conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)

        # res layer
        self.part_two = ConvModule(planes * 2, planes * 2, kernel_size=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv5 = ConvModule(planes * 2, planes, kernel_size=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv6 = ConvModule(planes, planes * 2, kernel_size=3, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.bottle_neck = ConvModule(planes * 2, planes * 2, kernel_size=1, conv_cfg=conv_cfg,
                                      norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.transition = ConvModule(planes * 4, planes * 2, kernel_size=1, conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # csp part
        l = self.part_one(x)
        r1 = self.part_two(x)
        r2 = self.conv5(r1)
        r2 = self.conv6(r2)
        r = r1 + r2  # shortcut

        r = self.bottle_neck(r)

        x = torch.cat([l, r], dim=1)
        x = self.transition(x)

        return x


class CSPResLayer(nn.Module):

    def __init__(self,
                 block,
                 num_blocks,
                 inplanes,  # in channels
                 planes,  # out channels
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=dict(type="BN"),
                 act_cfg=dict(type='Mish')):
        super(CSPResLayer, self).__init__()
        if downsample:
            self.down = ConvModule(inplanes, planes, kernel_size=3, stride=2, padding=1, conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)
        # csp
        self.part_one = ConvModule(planes, inplanes, kernel_size=1, stride=1, conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.part_two = ConvModule(planes, inplanes, kernel_size=1, stride=1, conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.res_layer = make_res_layer(block, inplanes, num_blocks, conv_cfg=conv_cfg,
                                        norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.bottleneck = ConvModule(inplanes, inplanes, kernel_size=1, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)
        self.transition = ConvModule(planes, planes, kernel_size=1, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

    def forward(self, x):
        x = self.down(x)
        part_one = self.part_one(x)
        part_two = self.part_two(x)

        part_two = self.res_layer(part_two)
        part_two = self.bottleneck(part_two)

        x = torch.cat([part_one, part_two], dim=1)
        x = self.transition(x)
        return x


@BACKBONES.register_module()
class CSPDarkNet(nn.Module):
    arch_settings =(BasicBlock, (2, 8, 8, 4))

    def __init__(self,
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Mish'),
                 norm_eval=True,
                 zero_init_residual=True):
        super(CSPDarkNet, self).__init__()

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual

        self.block, stage_blocks = self.arch_settings

        self.inplanes = 64
        self.stem_layer = StemLayer(32, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.csp_res_layers = []
        for i, num_blocks in enumerate(stage_blocks):
            planes = self.inplanes * 2
            csp_res_layer = CSPResLayer(self.block,
                                        num_blocks,
                                        self.inplanes,
                                        planes,
                                        downsample=True,
                                        conv_cfg=conv_cfg,
                                        norm_cfg=norm_cfg,
                                        act_cfg=act_cfg)
            self.inplanes = planes
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, csp_res_layer)
            self.csp_res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.inplanes

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        # freeze corresponding stages.
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x):
        x = self.stem_layer(x)
        outs = []
        for i, layer_name in enumerate(self.csp_res_layers):
            csp_res_layer = getattr(self, layer_name)
            x = csp_res_layer(x)

            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super(CSPDarkNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

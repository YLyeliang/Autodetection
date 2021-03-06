import logging
import math

import torch.nn as nn
from .cspresnet import CSPResNet
from .cspresnet import Bottleneck as _Bottleneck
from mtcv.cnn import build_norm_layer, build_conv_layer, ConvModule

from ..builder import BACKBONES


class Bottleneck(_Bottleneck):
    def __init__(self,
                 inplanes,
                 planes,
                 groups=1,
                 base_width=4,
                 *args,
                 **kwargs):
        """ Bottleneck block for ResNext."""
        super(Bottleneck, self).__init__(inplanes, planes, *args, **kwargs)

        if groups == 1:  # if groups ==1,it's a resnet Bottleneck
            width = self.planes
        else:
            width = math.floor(self.planes * (base_width) / 64 * groups)

        # establish 3 batch norm layer,
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, width, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, width, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(self.norm_cfg, self.planes * self.expansion, postfix=3)

        # establish conv layer, inplanes are input channels.
        self.conv1 = build_conv_layer(self.conv_cfg, self.inplanes, width, kernel_size=1,
                                      stride=self.conv1_stirde, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(self.conv_cfg, width, width, kernel_size=3,
                                      stride=self.conv2_stride, padding=self.dilation,
                                      dilation=self.dilation, groups=groups, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.conv3 = build_conv_layer(self.conv_cfg, width, self.planes * self.expansion,
                                      kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)


def make_res_layer(block,
                   planes,
                   blocks,
                   dilation=1,
                   groups=1,
                   base_width=4,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   act_cfg=dict(type='ReLU')):
    layers = []
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(inplanes, planes, stride=1, dilation=dilation, groups=groups,
                  base_width=base_width, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )

    return nn.Sequential(*layers)


@BACKBONES.register_module()
class CSPResNeXt(CSPResNet):
    """ResNeXt backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        groups (int): Group of resnext.
        base_width (int): Base width of resnext.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """
    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, groups=1, base_width=4, **kwargs):
        super(CSPResNeXt, self).__init__(**kwargs)
        self.groups = groups
        self.base_width = base_width

        self.inplanes = 64
        self.res_layers = []
        self.csp_layers = nn.ModuleList()
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]  # default: (1,2,2,2)
            dilation = self.dilations[i]  # default: (1,1,1,1) norm conv. >1 then atrous.
            planes = 64 * 2 ** i
            # csp block
            if stride != 1 or self.inplanes != planes * self.block.expansion:
                downsample = nn.Sequential(
                    build_conv_layer(self.conv_cfg, self.inplanes, planes * self.block.expansion,
                                     kernel_size=1, stride=stride, bias=False),
                    build_norm_layer(self.norm_cfg, planes * self.block.expansion)[1]
                    # 0 is norm name, 1 is norm layer.
                )
            down_layer = self.block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample,
                                    groups=groups, base_width=base_width, conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            left_csp = ConvModule(planes * self.block.expansion, planes * self.block.expansion, kernel_size=1,
                                  stride=1, bias=False)
            # The concatenation has changed to sum.
            end_csp = ConvModule(planes * self.block.expansion, planes * self.block.expansion, kernel_size=1, stride=1,
                                 bias=False)
            self.csp_layers.append(nn.ModuleList([down_layer, left_csp, end_csp]))
            res_layer = make_res_layer(
                self.block,
                planes,
                num_blocks,
                dilation=dilation,
                groups=self.groups,
                base_width=self.base_width,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

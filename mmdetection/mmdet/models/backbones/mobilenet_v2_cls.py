#from __future__ import division
import logging
import math
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.plugins import GeneralizedAttention
from mmdet.ops import ContextBlock, DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer




def conv_bn_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn_relu(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

@BACKBONES.register_module
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.,out_feature_indices=(10, 13, 17),
            frozen_stages=-1,norm_eval=True,with_classifier=False):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # p, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 1],
            [6, 96, 3, 2],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        #self.num_stages = num_stages
        self.out_feature_indices = out_feature_indices
        self.frozen_stages = frozen_stages
        #self.conv_cfg = conv_cfg
        #self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_classifier = with_classifier
        self.num_classes = num_classes
        # self.inplanes = 32
        # self.lastplanes = 1280

        # building first layer
        #assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn_relu(3, input_channel, 2)]
        # building inverted residual blocks
        for p, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, p))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, p))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn_relu(input_channel, self.last_channel))
        #self.features.append(nn.AvgPool2d(input_size//32))
        # make it nn.Sequential
        self.feat_dim = self.last_channel
        if self.with_classifier:
            self._make_tail_layer() 
        self.features = nn.Sequential(*self.features)
        #self._freeze_stages()
        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, n_class),
        # )

        self._initialize_weights()

    def _make_tail_layer(self):
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feat_dim, self.num_classes))
    def forward(self, x):
        outs = []
        for i, mobilev2_layer in enumerate(self.features):
            x = mobilev2_layer(x)
            #print(x.shape)
            if i in self.out_feature_indices:
                outs.append(x)

        if self.with_classifier:
            x = self.avgpool(x)
            x = x.reshape(x.size(0), -1)
            x = self.classifier(x)
            return x
        else:
            return tuple(outs)
        # x = self.features(x)
        # x = x.view(-1, self.last_channel)
        # x = self.classifier(x)
        return x

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            print('frozon_stages')
            # m = self.features[0]
            # m.eval()
            # for param in m.parameters():
            #     param.requires_grad = False
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                # if isinstance(m, _BatchNorm):
                #     m.eval()
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        # frozen_stages = self.frozen_stages \
        #     if self.frozen_stages <= len(self.stage_blocks) \
        #     else len(self.stage_blocks)

        # for i in range(1, frozen_stages + 2):
        #     for j in range(*self.range_sub_modules[i - 1]):
        #         m = self.features[j]
        #         m.eval()
        #         for param in m.parameters():
        #             param.requires_grad = False
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
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def train(self, mode=True):
        super(MobileNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
# def mobilenet_v2(**kwargs):
#     model = MobileNetV2(**kwargs)
#     return model

# Copyright (c) 2023 ✨Challyfilio✨
import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np

from typing import List, Tuple, Union
from torch import Tensor

from loguru import logger
from .spfn import Adjustment

from ..builder import ROTATED_NECKS
from mmcv.runner import BaseModule, auto_fp16


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


class LayerExpansion(nn.Module):
    def __init__(self,
                 out_channels,
                 out_layers=6,
                 feature_map_size=[256, 128, 64, 32],
                 name="layerexpansion"):
        super(LayerExpansion, self).__init__()
        self._out_channels = out_channels
        self._out_layers = out_layers
        self._feature_map_size = feature_map_size
        self._name = name
        #     self._build()

        # def _build(self):
        self._bifusion_list = nn.ModuleList()
        for i in range(self._out_layers - 3):
            self._bifusion_list.append(InputBIFusion(out_channels=self._out_channels,
                                                     btm_shape=self._feature_map_size[i],
                                                     top_shape=self._feature_map_size[i + 1],
                                                     name=self._name + "_inbifusion" + str(i)))

    def forward(self, input_ts_list):
        l1, l2, l3, l4 = input_ts_list
        if self._out_layers == 3:
            return [l1, l2, l3]
        elif self._out_layers == 5:
            l1l2 = self._bifusion_list[0](l1, l2)
            l2l3 = self._bifusion_list[1](l2, l3)
            return [l1, l1l2, l2, l2l3, l3]
        elif self._out_layers == 6:  # challyfilio SFPN-7
            l1l2 = self._bifusion_list[0](l1, l2)
            l2l3 = self._bifusion_list[1](l2, l3)
            l3l4 = self._bifusion_list[2](l3, l4)
            return [l1, l1l2, l2, l2l3, l3, l3l4, l4]
        elif self._out_layers == 9:
            l1l2 = self._bifusion_list[0](l1, l2)
            l2l3 = self._bifusion_list[1](l2, l3)
            l1l1l2 = self._bifusion_list[2](l1, l1l2)
            l1l2l2 = self._bifusion_list[3](l1l2, l2)
            l2l2l3 = self._bifusion_list[4](l2, l2l3)
            l2l3l3 = self._bifusion_list[5](l2l3, l3)
            return [l1, l1l1l2, l1l2, l1l2l2, l2, l2l2l3, l2l3, l2l3l3, l3]


class InputBIFusion(nn.Module):
    def __init__(self, out_channels, btm_shape, top_shape, name="inputbufusion"):
        super(InputBIFusion, self).__init__()
        # self._btm_down = None
        # self._top_up = None
        # self._conv = None
        self._out_channels = out_channels
        self._btm_shape = btm_shape
        self._top_shape = top_shape
        self._name = name

        # def _build(self, btm_shape, top_shape):
        #     btm_shape = np.array(btm_shape)
        #     top_shape = np.array(top_shape)
        self._target_shape = np.round((self._btm_shape + self._top_shape) / 2).astype(int)
        # self._btm_down=AdaptPooling(list(target_shape[1:]))
        # self._top_up=AdaptUpsample(list(target_shape[1:]))
        self._btm_down = nn.AdaptiveAvgPool2d((self._target_shape, self._target_shape))
        self._top_up = nn.Upsample(size=(self._target_shape, self._target_shape), mode='bilinear', align_corners=True)
        self._conv = nn.Sequential(
            nn.Conv2d(in_channels=self._out_channels, out_channels=self._out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_features=self._out_channels),
            Mish()
        )

    def forward(self, btm_ts, top_ts):
        # btm_shape = list(btm_ts.shape)[1:]
        # top_shape = list(top_ts.shape)[1:]
        # self._build(btm_shape, top_shape)
        btm_down = self._btm_down(btm_ts)
        top_up = self._top_up(top_ts)
        x = btm_down + top_up
        output_ts = self._conv(x)
        return output_ts


class AdaptPooling(nn.Module):
    def __init__(self, output_hw, name="adaptpooling"):
        super(AdaptPooling, self).__init__()
        self._output_hw = output_hw
        self._name = name

    def forward(self, input_ts):
        output_ts = F.interpolate(input_ts, size=self._output_hw, mode='bilinear', align_corners=True)
        return output_ts


class AdaptUpsample(nn.Module):
    def __init__(self, output_hw, name="adaptupsample"):
        super(AdaptUpsample, self).__init__()
        self._output_hw = output_hw
        self._name = name

    def forward(self, input_ts):
        output_ts = F.interpolate(input_ts, size=self._output_hw, mode='bilinear', align_corners=True)
        return output_ts


class ConvBN(nn.Module):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding=1,
                 bias=False,
                 use_bn=True,
                 activation=None,
                 name="convbn"):
        super(ConvBN, self).__init__()
        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._bias = bias
        self._use_bn = use_bn
        self._activation = activation
        self._name = name
        #     self._Build()

        # def _Build(self):
        self._conv = nn.Conv2d(in_channels=self._filters,
                               out_channels=self._filters,
                               kernel_size=self._kernel_size,
                               stride=self._strides,
                               padding=self._padding,
                               bias=self._bias)
        if self._use_bn:
            self._bn = nn.BatchNorm2d(num_features=self._filters, momentum=0.997, eps=1e-4)
        if self._activation is not None:
            if self._activation == 'relu':
                self._act = nn.ReLU()
            elif self._activation == 'sigmoid':
                self._act = nn.Sigmoid()
            elif self._activation == 'mish':
                self._act = Mish()
            else:
                raise ValueError(f'Unsupported activation function: {self._activation}')

    def forward(self, input_ts):
        x = self._conv(input_ts)
        if self._use_bn:
            x = self._bn(x)
        if hasattr(self, '_act'):
            x = self._act(x)
        output_ts = x
        return output_ts


class BIFusion(nn.Module):  # SFM
    def __init__(self, out_channels, mid_shape, name="bufusion"):
        super(BIFusion, self).__init__()
        self._out_channels = out_channels
        self._mid_shape = (mid_shape, mid_shape)  # (,)
        self._name = name

        # def _Build(self,mid_shape):
        # target_shape=mid_shape
        self._btm_down = AdaptPooling(self._mid_shape, name=self._name + "_btm_down")
        self._top_up = AdaptUpsample(self._mid_shape, name=self._name + "_top_up")
        self._conv = ConvBN(filters=self._out_channels,
                            kernel_size=(3, 3),
                            activation='mish',
                            name=self._name + "_conv")

    def forward(self, btm_ts, mid_ts, top_ts):
        # mid_shape = list(mid_ts.shape)[1:]
        # self._Build(mid_shape)
        out_ts = mid_ts
        if (btm_ts != None):
            btm_down = self._btm_down(btm_ts)
            out_ts = out_ts + btm_down
        if (top_ts != None):
            top_up = self._top_up(top_ts)
            out_ts = out_ts + top_up
        output_ts = self._conv(out_ts)
        return output_ts


class FusionPhase1(nn.Module):  # SFB1
    def __init__(self,
                 out_channels,
                 mid_shape_list=[192, 96, 48],
                 input_ts_len=7,
                 name="fusionphase1"):
        super(FusionPhase1, self).__init__()
        self._out_channels = out_channels
        self._mid_shape_list = mid_shape_list
        self._input_ts_len = input_ts_len
        self._name = name

        # def _Build(self,input_ts_len):
        self._bidusion_list = nn.ModuleList()  # https://github.com/open-mmlab/mmdetection/issues/9725
        for i in range(self._input_ts_len - 4):
            self._bidusion_list.append(BIFusion(out_channels=self._out_channels,
                                                mid_shape=self._mid_shape_list[i],
                                                name=self._name + "_bifusion" + str(i)))

    def forward(self, input_ts_list: List[Tensor]) -> list:
        # input_ts_len=len(input_ts_list)
        # self._Build(input_ts_len)
        if (self._input_ts_len == 3):
            l1, l2, l3 = input_ts_list
            l2 = self._bidusion_list[0](l1, l2, l3)
            return [l1, l2, l3]
        elif (self._input_ts_len == 5):
            l1, l2, l3, l4, l5 = input_ts_list
            l2 = self._bidusion_list[0](l1, l2, l3)
            l4 = self._bidusion_list[1](l3, l4, l5)
            return [l1, l2, l3, l4, l5]
        elif (self._input_ts_len == 7):  # challyfilio SFPN-7
            l1, l2, l3, l4, l5, l6, l7 = input_ts_list
            l2 = self._bidusion_list[0](l1, l2, l3)
            l4 = self._bidusion_list[1](l3, l4, l5)
            l6 = self._bidusion_list[2](l5, l6, l7)
            return [l1, l2, l3, l4, l5, l6, l7]
        elif (self._input_ts_len == 9):
            l1, l2, l3, l4, l5, l6, l7, l8, l9 = input_ts_list
            l2 = self._bidusion_list[0](l1, l2, l3)
            l4 = self._bidusion_list[1](l3, l4, l5)
            l6 = self._bidusion_list[2](l4, l6, l7)  # 我发现这里写错了 哈哈哈哈哈哈 垃圾 这也好意思公开代码 l4->l5
            l8 = self._bidusion_list[3](l7, l8, l9)
            return [l1, l2, l3, l4, l5, l6, l7, l8, l9]


class FusionPhase2(nn.Module):  # SFB2
    def __init__(self,
                 out_channels,
                 mid_shape_list=[256, 128, 64, 32],
                 input_ts_len=7,
                 name="fusionphase2"):
        super(FusionPhase2, self).__init__()
        self._out_channels = out_channels
        self._mid_shape_list = mid_shape_list
        self._input_ts_len = input_ts_len
        self._name = name

        # def _Build(self,input_ts_len):
        self._bidusion_list = nn.ModuleList()  # https://github.com/open-mmlab/mmdetection/issues/9725
        for i in range(input_ts_len - 3):
            self._bidusion_list.append(BIFusion(out_channels=self._out_channels,
                                                mid_shape=self._mid_shape_list[i],
                                                name=self._name + "_bifusion" + str(i)))

    def forward(self, input_ts_list: List[Tensor]) -> list:
        # input_ts_len=len(input_ts_list)
        # self._Build(input_ts_len)
        if (self._input_ts_len == 3):
            l1, l2, l3 = input_ts_list
            l1 = self._bidusion_list[0](None, l1, l2)
            l3 = self._bidusion_list[1](l2, l3, None)
            return [l1, l2, l3]
        elif (self._input_ts_len == 5):
            l1, l2, l3, l4, l5 = input_ts_list
            l1 = self._bidusion_list[0](None, l1, l2)
            l3 = self._bidusion_list[1](l2, l3, l4)
            l5 = self._bidusion_list[2](l4, l5, None)
            return [l1, l2, l3, l4, l5]
        elif (self._input_ts_len == 7):  # challyfilio SFPN-7
            l1, l2, l3, l4, l5, l6, l7 = input_ts_list
            l1 = self._bidusion_list[0](None, l1, l2)
            l3 = self._bidusion_list[1](l2, l3, l4)
            l5 = self._bidusion_list[2](l4, l5, l6)
            l7 = self._bidusion_list[3](l6, l7, None)
            return [l1, l2, l3, l4, l5, l6, l7]
        elif (self._input_ts_len == 9):
            l1, l2, l3, l4, l5, l6, l7, l8, l9 = input_ts_list
            l1 = self._bidusion_list[0](None, l1, l2)
            l3 = self._bidusion_list[1](l2, l3, l4)
            l5 = self._bidusion_list[2](l4, l5, l6)
            l7 = self._bidusion_list[3](l6, l7, l8)
            l9 = self._bidusion_list[4](l8, l9, None)
            return [l1, l2, l3, l4, l5, l6, l7, l8, l9]


class SFBs(nn.Module):
    def __init__(self,
                 out_channels,
                 mid_shape_list_1,
                 mid_shape_list_2,
                 repeat=3,
                 name="sfbs") -> None:
        super(SFBs, self).__init__()
        self._out_channels = out_channels
        self._mid_shape_list_1 = mid_shape_list_1
        self._mid_shape_list_2 = mid_shape_list_2
        self._repeat = repeat
        self._name = name

        self._fusion_phase1_list = nn.ModuleList()  # https://github.com/open-mmlab/mmdetection/issues/9725
        self._fusion_phase2_list = nn.ModuleList()  # https://github.com/open-mmlab/mmdetection/issues/9725
        for i in range(self._repeat):
            self._fusion_phase1_list.append(FusionPhase1(out_channels=self._out_channels,
                                                         mid_shape_list=self._mid_shape_list_1,
                                                         name=self._name + "_phase1_" + str(i)))
            self._fusion_phase2_list.append(FusionPhase2(out_channels=self._out_channels,
                                                         mid_shape_list=self._mid_shape_list_2,
                                                         name=self._name + "_phase2_" + str(i)))

    def forward(self, input_ts_list: List[Tensor]) -> list:
        out_ts_list = input_ts_list
        for i in range(self._repeat):
            last_out_ts_list = out_ts_list.copy()
            out_ts_list = self._fusion_phase1_list[i](out_ts_list)
            out_ts_list = self._fusion_phase2_list[i](out_ts_list)
            for ts_idx in range(len(out_ts_list)):
                out_ts_list[ts_idx] = out_ts_list[ts_idx] + last_out_ts_list[ts_idx]
        return out_ts_list


class InputAdjSize(nn.Module):
    def __init__(self, target_shape, name="inputadjsize"):
        super(InputAdjSize, self).__init__()
        self._target_shape = target_shape
        self._name = name
        self._adj = nn.Upsample(size=(self._target_shape, self._target_shape), mode='bilinear', align_corners=True)

    def forward(self, adj):
        adj_out = self._adj(adj)
        return adj_out


@ROTATED_NECKS.register_module()
class CSLFPN(nn.Module):
    def __init__(
            self,
            feature_map_shape: List[int],
            in_channels: List[int],
            out_channels: int,
            num_outs: int,
            add_extra_convs: Union[bool, str] = False
    ) -> None:
        super(CSLFPN, self).__init__()
        assert isinstance(feature_map_shape, list)
        assert isinstance(in_channels, list)
        self.feature_map_shape = feature_map_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.num_ins = len(in_channels)
        self.add_extra_convs = add_extra_convs

        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.adjust2 = Adjustment(self.in_channels[0], self.out_channels)
        self.adjust3 = Adjustment(self.in_channels[1], self.out_channels)
        self.adjust4 = Adjustment(self.in_channels[2], self.out_channels)
        self.adjust5 = Adjustment(self.in_channels[3], self.out_channels)

        self.mid_shape_list_1 = []
        for i in range(len(self.feature_map_shape) - 1):
            avg = int((self.feature_map_shape[i] + self.feature_map_shape[i + 1]) / 2)
            self.mid_shape_list_1.append(avg)

        self.LayerExpansion = LayerExpansion(out_channels=self.out_channels,
                                             feature_map_size=self.feature_map_shape)
        self.SFBs = SFBs(out_channels=self.out_channels,
                         mid_shape_list_1=self.mid_shape_list_1,
                         mid_shape_list_2=self.feature_map_shape)

        self._adjsize_list = nn.ModuleList()
        for i in self.feature_map_shape:
            self._adjsize_list.append(InputAdjSize(target_shape=i))
        # self._top_up = nn.Upsample(size=(self._target_shape, self._target_shape), mode='bilinear', align_corners=True)

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        # x2, x3, x4, x5 = inputs

        outs = list(inputs)
        for i in range(len(outs)):
            outs[i] = self._adjsize_list[i](outs[i])
            # outs[i] = F.interpolate(outs[i], size=self.feature_map_shape[i], mode='bilinear', align_corners=True)
        x2, x3, x4, x5 = outs

        l1 = self.adjust2(x2)
        l2 = self.adjust3(x3)
        l3 = self.adjust4(x4)
        l4 = self.adjust5(x5)

        input_list = self.LayerExpansion([l1, l2, l3, l4])
        output_list = self.SFBs(input_list)
        outs = [output_list[0], output_list[2], output_list[4], output_list[6]]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # Faster R-CNN
            if not self.add_extra_convs:
                for i in range(self.num_outs - self.num_ins):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)


if __name__ == "__main__":
    input_tensor2 = torch.rand(2, 96, 256, 256)
    input_tensor3 = torch.rand(2, 192, 128, 128)
    input_tensor4 = torch.rand(2, 384, 64, 64)
    input_tensor5 = torch.rand(2, 768, 32, 32)
    channels50 = [256, 512, 1024, 2048]  # rn50
    channels34 = [64, 128, 256, 512]  # rn34
    channels_s = [96, 192, 384, 768]
    model = CSLFPN(feature_map_shape=[256, 128, 64, 32],
                   in_channels=channels_s,
                   out_channels=256,
                   num_outs=4)
    outputs = model((input_tensor2, input_tensor3, input_tensor4, input_tensor5))
    logger.success(len(outputs))
    for j in outputs:
        print(j.shape)
    exit()

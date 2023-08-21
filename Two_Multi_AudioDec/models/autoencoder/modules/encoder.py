#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://ieeexplore.ieee.org/document/9625818)

"""Encoder modules."""

import torch
import inspect

from layers.conv_layer import NonCausalConv1d
from layers.conv_layer import CausalConv1d
from models.autoencoder.modules.residual_unit import NonCausalResidualUnit
from models.autoencoder.modules.residual_unit import CausalResidualUnit
from models.utils import check_mode


class EncoderBlock(torch.nn.Module):
    """ Encoder block (downsampling) """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        dilations=(1, 3, 9),
        bias=True,
        mode='causal',
    ):
        super().__init__()
        self.mode = mode
        if self.mode == 'noncausal':
            ResidualUnit = NonCausalResidualUnit
            Conv1d = NonCausalConv1d
        elif self.mode == 'causal':
            ResidualUnit = CausalResidualUnit
            Conv1d = CausalConv1d
        else:
            raise NotImplementedError(f"Mode ({self.mode}) is not supported!")

        self.res_units = torch.nn.ModuleList()
        for dilation in dilations:
            self.res_units += [
                ResidualUnit(in_channels, in_channels, dilation=dilation)]
        self.num_res = len(self.res_units)

        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(2 * stride),
            stride=stride,
            bias=bias,
        )
        
    def forward(self, x):
        for idx in range(self.num_res):
            x = self.res_units[idx](x)
        x = self.conv(x)
        return x
    
    def inference(self, x):
        check_mode(self.mode, inspect.stack()[0][3])
        for idx in range(self.num_res):
            x = self.res_units[idx].inference(x)
        x = self.conv.inference(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self,
        input_channels,
        encode_channels,
        combine_channel_ratios=(2, 4),
        seperate_channel_ratios_speech=(8, 16,32),
        seperate_channel_ratios_rir=(8, 8, 16,32),
        combine_strides=(2, 2),
        seperate_strides_speech=(3, 5, 5),
        seperate_strides_rir=(2, 3, 5, 5, 5),
        kernel_size=7,
        bias=True,
        mode='causal',
    ):
        super().__init__()
        assert len(combine_channel_ratios) == len(combine_strides)
        assert len(seperate_channel_ratios_speech) == len(seperate_strides_speech)
        assert len(seperate_channel_ratios_rir) == len(seperate_strides_rir)
        self.mode = mode
        if self.mode == 'noncausal':
            Conv1d = NonCausalConv1d
        elif self.mode == 'causal':
            Conv1d = CausalConv1d
        else:
            raise NotImplementedError(f"Mode ({self.mode}) is not supported!")

        self.conv = Conv1d(
            in_channels=input_channels, 
            out_channels=encode_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            bias=False)

        self.conv_combine = Conv1d(
            in_channels=input_channels, 
            out_channels=input_channels, 
            kernel_size=3, 
            stride=1, 
            bias=False)

        self.combine_conv_blocks = torch.nn.ModuleList()
        self.seperate_conv_blocks_1 = torch.nn.ModuleList()
        # self.seperate_conv_blocks_2 = torch.nn.ModuleList()
        


        in_channels = encode_channels

        self.encode_RIR = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, 8*in_channels, 96001, 1500, 48000, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 1024
            torch.nn.Conv1d(8*in_channels, 16*in_channels, 41, 2, 20, bias=False),
            torch.nn.BatchNorm1d(16*in_channels),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(16*in_channels, 32*in_channels, 41, 2, 20, bias=False),
            torch.nn.BatchNorm1d(32*in_channels),
            torch.nn.LeakyReLU(0.2, inplace=True),
            
        )

        self.encode_RIR_s = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 16*in_channels, 96001, 3000, 48000, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 1024
            torch.nn.Conv1d(16*in_channels, 32*in_channels, 41, 2, 20, bias=False),
            torch.nn.BatchNorm1d(32*in_channels),
            torch.nn.LeakyReLU(0.2, inplace=True),
            
        )

        for idx, stride in enumerate(combine_strides):
            out_channels = encode_channels * combine_channel_ratios[idx]
            self.combine_conv_blocks += [
                EncoderBlock(in_channels, out_channels, stride, bias=bias, mode=self.mode)]
            in_channels = out_channels

        seperate_in_channels =  in_channels

        in_channels=seperate_in_channels

        for idx, stride in enumerate(seperate_strides_speech):
            out_channels = encode_channels * seperate_channel_ratios_speech[idx]
            self.seperate_conv_blocks_1 += [
                EncoderBlock(in_channels, out_channels, stride, bias=bias, mode=self.mode)]
            in_channels = out_channels

        in_channels=seperate_in_channels

        # for idx, stride in enumerate(seperate_strides_rir):
        #     out_channels = encode_channels * seperate_channel_ratios_rir[idx]
        #     self.seperate_conv_blocks_2 += [
        #         EncoderBlock(in_channels, out_channels, stride, bias=bias, mode=self.mode)]
        #     in_channels = out_channels

        self.combine_num_blocks = len(self.combine_conv_blocks)
        self.seperate_num_blocks_speech = len(self.seperate_conv_blocks_1)
        # self.seperate_num_blocks_rir = len(self.seperate_conv_blocks_2)
        self.out_channels = out_channels
    
    def forward(self, x): 
        x_combine = self.conv_combine(x)

        x_speech = self.conv(x_combine)
        # for i in range(self.combine_num_blocks):
        #     x_combine = self.combine_conv_blocks[i](x_combine)

        # x_speech = x_combine
        x_rir = x_combine

        for i in range(self.seperate_num_blocks_speech):
            x_speech = self.seperate_conv_blocks_1[i](x_speech)
        # x_speech = self.encode_RIR_s(x_speech)
        # for i in range(self.seperate_num_blocks_rir):
        #     x_rir = self.seperate_conv_blocks_2[i](x_rir)
        x_rir = self.encode_RIR(x_rir)


        return x_speech, x_rir
    
    def encode(self, x):
        check_mode(self.mode, inspect.stack()[0][3])
        x_combine = self.conv_combine(x)

        x_speech = self.conv(x_combine)
        # for i in range(self.combine_num_blocks):
        #     x_combine = self.combine_conv_blocks[i](x_combine)

        # x_speech = x_combine
        x_rir = x_combine

        for i in range(self.seperate_num_blocks_speech):
            x_speech = self.seperate_conv_blocks_1[i](x_speech)
        # x_speech = self.encode_RIR_s(x_speech)
        # for i in range(self.seperate_num_blocks_rir):
        #     x_rir = self.seperate_conv_blocks_2[i](x_rir)
        x_rir = self.encode_RIR(x_rir)


        return x_speech, x_rir


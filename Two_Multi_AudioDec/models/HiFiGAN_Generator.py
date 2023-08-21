#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""AudioDec model."""

import torch
import inspect

from layers.conv_layer import CausalConv1d, CausalConvTranspose1d
from models.vocoder.HiFiGAN import Generator as generator_hifigan
from models.autoencoder.modules.decoder_rir import Decoder_RIR
from models.utils import check_mode


### GENERATOR ###
class Generator(torch.nn.Module):
    """AudioDec generator."""

    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        channels=512,
        kernel_size=7,
        upsample_scales=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        groups=1,
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        stats=None,
        output_channels_rir=2, #
        decode_channels=16, #
        code_dim=64, #
        # bias=True, #
        rir_dec_ratios=(128, 64, 32, 16, 8, 4), #
        rir_dec_strides=(5, 5, 5, 3, 2, 2), #
        mode='causal', #
        codec='audiodec',
    ):
        super().__init__()
        if codec == 'audiodec':
            decoder_speech = generator_hifigan
            decoder_rir = Decoder_RIR

        else:
            raise NotImplementedError(f"Codec ({codec}) is not supported!")
        self.mode = mode
        



        self.decoder_speech = decoder_speech(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            kernel_size=kernel_size,
            upsample_scales=upsample_scales,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilations=resblock_dilations,
            groups=groups,
            bias=bias,
            use_additional_convs=use_additional_convs,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            use_weight_norm=use_weight_norm,
            stats=stats,
        )


        self.decoder_rir = decoder_rir(
            code_dim=code_dim,
            output_channels=output_channels_rir,
            decode_channels=decode_channels,
            channel_ratios=rir_dec_ratios,
            strides=rir_dec_strides,
            kernel_size=7,
            bias=bias,
            mode=self.mode,
        )

      



    def forward(self, zq_speech, zq_rir):
        

        y_speech = self.decoder_speech(zq_speech)
        y_rir = self.decoder_rir(zq_rir)



        return y_speech, y_rir



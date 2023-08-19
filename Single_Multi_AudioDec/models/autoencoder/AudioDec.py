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
from models.autoencoder.modules.encoder import Encoder
from models.autoencoder.modules.decoder_rir import Decoder_RIR
from models.autoencoder.modules.decoder_speech import Decoder_SPEECH
from models.autoencoder.modules.projector import Projector
from models.autoencoder.modules.quantizer import Quantizer
from models.utils import check_mode


### GENERATOR ###
class Generator(torch.nn.Module):
    """AudioDec generator."""

    def __init__(
        self,
        input_channels=2,
        output_channels_rir=2,
        output_channels_speech=1,
        encode_channels=16,
        decode_channels=16,
        code_dim=64,
        codebook_num=8,
        codebook_size=1024,
        bias=True,
        combine_enc_ratios=(2,  4),
        seperate_enc_ratios_speech=(8, 16, 32),
        seperate_enc_ratios_rir=(8, 12, 16, 32),
        speech_dec_ratios=(32, 16, 8, 4, 2),
        rir_dec_ratios=(128, 64, 32, 16, 8, 4),
        combine_enc_strides=(2, 2),
        seperate_enc_strides_speech=(3, 5, 5),
        seperate_enc_strides_rir=(3, 5, 5, 5),
        speech_dec_strides=(5, 5, 3, 2, 2),
        rir_dec_strides=(5, 5, 5, 3, 2, 2),
        mode='causal',
        codec='audiodec',
        projector='conv1d',
        quantier='residual_vq',
    ):
        super().__init__()
        if codec == 'audiodec':
            encoder = Encoder
            decoder_speech = Decoder_SPEECH
            decoder_rir = Decoder_RIR

        else:
            raise NotImplementedError(f"Codec ({codec}) is not supported!")
        self.mode = mode
        self.input_channels = input_channels

        self.encoder = encoder(
            input_channels=input_channels,
            encode_channels=encode_channels,
            combine_channel_ratios=combine_enc_ratios,
            seperate_channel_ratios_speech=seperate_enc_ratios_speech,
            seperate_channel_ratios_rir=seperate_enc_ratios_rir,
            combine_strides=combine_enc_strides,
            seperate_strides_speech=seperate_enc_strides_speech,
            seperate_strides_rir=seperate_enc_strides_rir,
            kernel_size=7,
            bias=bias,
            mode=self.mode,
        )

        self.decoder_speech = decoder_speech(
            code_dim=code_dim,
            output_channels=output_channels_speech,
            decode_channels=decode_channels,
            channel_ratios=speech_dec_ratios,
            strides=speech_dec_strides,
            kernel_size=7,
            bias=bias,
            mode=self.mode,
        )

        self.projector_speech = Projector(
            input_channels=self.encoder.out_channels,
            code_dim=code_dim,
            kernel_size=3,
            stride=1,
            bias=False,
            mode=self.mode,
            model=projector,
        )

        self.quantizer_speech = Quantizer(
            code_dim=code_dim,
            codebook_num=codebook_num,
            codebook_size=codebook_size,
            model=quantier,
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

        self.projector_rir = Projector(
            input_channels=self.encoder.out_channels,
            code_dim=code_dim,
            kernel_size=3,
            stride=1,
            bias=False,
            mode=self.mode,
            model=projector,
        )

        self.quantizer_rir = Quantizer(
            code_dim=code_dim,
            codebook_num=codebook_num,
            codebook_size=codebook_size,
            model=quantier,
        )



    def forward(self, x):
        (batch, channel, length) = x.size()
        if channel != self.input_channels: 
            x = x.reshape(-1, self.input_channels, length) # (B, C, T) -> (B', C', T)
        x_speech, x_rir = self.encoder(x)

        # print("x shape  ",x.shape)
        # print("x_speech shape  ",x_speech.shape)
        # print("x_rir shape  ",x_rir.shape)


        z_speech = self.projector_speech(x_speech)
        zq_speech, vqloss_speech, perplexity_speech = self.quantizer_speech(z_speech)

        # print("z_speech shape  ",z_speech.shape)
        # print("zq_speech shape  ",zq_speech.shape)

        z_rir = self.projector_rir(x_rir)
        zq_rir, vqloss_rir, perplexity_rir = self.quantizer_rir(z_rir)

        # print("z_rir shape  ",z_rir.shape)
        # print("zq_rir shape  ",zq_rir.shape)
        # print("zq_rir shape 1 ",zq_rir.shape[1])


        y_speech = self.decoder_speech(zq_speech)
        y_rir = self.decoder_rir(zq_rir)



        return y_speech, y_rir, zq_speech, zq_rir, z_speech, z_rir, vqloss_speech, vqloss_rir, perplexity_speech, perplexity_rir


#########################Will change this later##########################################
# STREAMING
# class StreamGenerator(Generator):
#     """AudioDec streaming generator."""

#     def __init__(
#         self,
#         input_channels=1,
#         output_channels=1,
#         encode_channels=32,
#         decode_channels=32,
#         code_dim=64,
#         codebook_num=8,
#         codebook_size=1024,
#         bias=True,
#         enc_ratios=(2, 4, 8, 16),
#         dec_ratios=(16, 8, 4, 2),
#         enc_strides=(3, 4, 5, 5),
#         dec_strides=(5, 5, 4, 3),
#         mode='causal',
#         codec='audiodec',
#         projector='conv1d',
#         quantier='residual_vq',
#     ):
#         super(StreamGenerator, self).__init__(
#             input_channels=input_channels,
#             output_channels=output_channels,
#             encode_channels=encode_channels,
#             decode_channels=decode_channels,
#             code_dim=code_dim,
#             codebook_num=codebook_num,
#             codebook_size=codebook_size,
#             bias=bias,
#             enc_ratios=enc_ratios,
#             dec_ratios=dec_ratios,
#             enc_strides=enc_strides,
#             dec_strides=dec_strides,
#             mode=mode,
#             codec=codec,
#             projector=projector,
#             quantier=quantier,
#         )
#         check_mode(mode, "AudioDec Streamer")
#         self.reset_buffer()


#     def initial_encoder(self, receptive_length, device):
#         self.quantizer.initial()
#         z = self.encode(torch.zeros(1, self.input_channels, receptive_length).to(device))
#         idx = self.quantize(z)
#         zq = self.lookup(idx)
#         return zq


#     def initial_decoder(self, zq):
#         self.decode(zq)


#     def encode(self, x):
#         (batch, channel, length) = x.size()
#         if channel != self.input_channels: 
#             x = x.reshape(-1, self.input_channels, length) # (B, C, T) -> (B', C', T)
#         x = self.encoder.encode(x)
#         z = self.projector.encode(x)
#         return z


#     def quantize(self, z):
#         zq, idx = self.quantizer.encode(z)
#         return idx


#     def lookup(self, idx):
#         return self.quantizer.decode(idx)


#     def decode(self, zq):
#         return self.decoder.decode(zq.transpose(2, 1))


#     def reset_buffer(self):
#         """Apply weight normalization module from all layers."""

#         def _reset_buffer(m):
#             if isinstance(m, CausalConv1d) or isinstance(m, CausalConvTranspose1d):
#                 m.reset_buffer()
#         self.apply(_reset_buffer)

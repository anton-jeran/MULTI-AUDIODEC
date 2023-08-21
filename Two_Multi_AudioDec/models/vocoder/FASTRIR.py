#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)
# Reference (https://github.com/jik876/hifi-gan)

"""HiFi-GAN Modules. (Causal)"""

import torch
import torch.nn as nn
import torch.nn.parallel
# from miscc.config import cfg
from torch.autograd import Variable





class Discriminator_RIR(nn.Module):
    def __init__(self,
        dis_dim = 96):
        super(Discriminator_RIR, self).__init__()
        self.df_dim  = dis_dim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        kernel_length =41
        self.encode_RIR = nn.Sequential(
            nn.Conv1d(2, ndf, kernel_length, 6, 20, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 1024
            nn.Conv1d(ndf, ndf * 2, kernel_length, 5, 20, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 256
            nn.Conv1d(ndf*2, ndf * 4, kernel_length, 5, 20, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size (ndf*4) x 64
            nn.Conv1d(ndf*4, ndf * 8, kernel_length, 5, 20, bias=False),
            nn.BatchNorm1d(ndf * 8),
            # state size (ndf * 8) x 16)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf*8, ndf * 16, kernel_length, 4, 20, bias=False),
            nn.BatchNorm1d(ndf * 16),
            # state size (ndf * 8) x 16)
            nn.LeakyReLU(0.2, inplace=True)
        )

        # self.encode_RIR1 = nn.Sequential(
        #     nn.Conv1d(2, ndf, kernel_length, 6, 20, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True))
        #     # state size. (ndf) x 1024
        # self.encode_RIR2 = nn.Sequential(
        #     nn.Conv1d(ndf, ndf * 2, kernel_length, 5, 20, bias=False),
        #     nn.BatchNorm1d(ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True))
        #     # state size (ndf*2) x 256
        # self.encode_RIR3 = nn.Sequential(
        #     nn.Conv1d(ndf*2, ndf * 4, kernel_length, 5, 20, bias=False),
        #     nn.BatchNorm1d(ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True))
        #     # # state size (ndf*4) x 64
        # self.encode_RIR4 = nn.Sequential(
        #     nn.Conv1d(ndf*4, ndf * 8, kernel_length, 5, 20, bias=False),
        #     nn.BatchNorm1d(ndf * 8),
        #     # state size (ndf * 8) x 16)
        #     nn.LeakyReLU(0.2, inplace=True))
        # self.encode_RIR5 = nn.Sequential(
        #     nn.Conv1d(ndf*8, ndf * 16, kernel_length,4 , 20, bias=False),
        #     nn.BatchNorm1d(ndf * 16),
        #     # state size (ndf * 8) x 16)
        #     nn.LeakyReLU(0.2, inplace=True)
        # )

        self.convd1d =  nn.ConvTranspose1d(ndf*16,ndf //2,kernel_size=kernel_length,stride=1, padding=20)

        self.outlogits = nn.Sequential(
                nn.Conv1d(ndf // 2 , 1, kernel_size=16, stride=4),
                # nn.Conv1d(1, 1, kernel_size=16, stride=4),
                nn.Sigmoid())



    def forward(self, RIRs):
        
        RIR_embedding = self.encode_RIR(RIRs)
        # RIR_embedding = self.encode_RIR2(RIR_embedding)
        # RIR_embedding = self.encode_RIR3(RIR_embedding)
        # RIR_embedding = self.encode_RIR4(RIR_embedding)
        # RIR_embedding = self.encode_RIR5(RIR_embedding)
        RIR_embedding = self.convd1d(RIR_embedding)
        RIR_embedding = self.outlogits(RIR_embedding)
       
        return RIR_embedding


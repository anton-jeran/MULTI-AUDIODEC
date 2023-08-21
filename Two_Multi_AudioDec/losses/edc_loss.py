#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Waveform-based loss modules."""

import torch
import cupy as cp
from cupyx.scipy.signal import fftconvolve
##########Check Code in GAMMA machine and implement this ###################
class EnergyDecayCurveLoss(torch.nn.Module):
    """Waveform shape loss."""

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()
        

    def forward(self, y_hat, y,filters):
        """Calculate MSE loss.

        Args:
            y_hat (Tensor): Generated single tensor (B, 1, T).
            y (Tensor): Groundtruth single tensor (B, 1, T).

        Returns:
            Tensor: MSE loss value.

        """
        filter_length = 16384  # a magic number, not need to tweak this much
        mult1 = 10

        real_ec = convert_IR2EC_batch(cp.asarray(y), filters, filter_length)
        fake_ec = convert_IR2EC_batch(cp.asarray(y_hat.to("cpu").detach()), filters, filter_length)



        divergence_loss = self.loss(real_ec,fake_ec) * mult1
        
        # loss = self.loss(y_hat, y)*(y.shape[2])
        
        return divergence_loss


def convert_IR2EC(rir, filters, filter_length):
    subband_ECs = np.zeros((len(rir), filters.shape[1]))
    for i in range(filters.shape[1]):
        subband_ir = scipy.signal.fftconvolve(rir, filters[:, i])
        subband_ir = subband_ir[(filter_length - 1):]
        squared = np.square(subband_ir[:len(rir)])
        subband_ECs[:, i] = np.cumsum(squared[::-1])[::-1]
    return subband_ECs

def convert_IR2EC_batch(rir, filters, filter_length):
    # filters = cp.asarray([[filters]])
    rir = rir[:,:,0:3968]
    subband_ECs = cp.zeros((rir.shape[0],rir.shape[1],rir.shape[2], filters.shape[3]))
    for i in range(filters.shape[3]):
        subband_ir = fftconvolve(rir, filters[:,:,:, i])
        subband_ir = subband_ir[:,:,(filter_length - 1):]
        squared = cp.square(subband_ir[:,:,:rir.shape[2]])
        subband_ECs[:, :,:,i] = cp.log(cp.cumsum(squared[:,:,::-1],axis=2)[:,:,::-1])
    subband_ECs = torch.tensor(subband_ECs,device='cuda')
    return subband_ECs    

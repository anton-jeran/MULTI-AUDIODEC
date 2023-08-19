#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Waveform-based loss modules."""

import torch
from torch import nn
from torch_stoi import NegSTOILoss

class STOILoss(torch.nn.Module):
    """Waveform shape loss."""

    def __init__(self):
        super().__init__()
        self.loss_func = NegSTOILoss(sample_rate=48000)
        

    def forward(self, y_hat, y):
        """Calculate MSE loss.

        Args:
            y_hat (Tensor): Generated single tensor (B, 1, T).
            y (Tensor): Groundtruth single tensor (B, 1, T).

        Returns:
            Tensor: MSE loss value.

        """
        
        loss = self.loss_func(y_hat, y).squeeze()
        
        loss = loss.mean()
        
        
        return loss
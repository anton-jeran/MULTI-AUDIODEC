#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Waveform-based loss modules."""

import torch


class TimeDomainMSELoss(torch.nn.Module):
    """Waveform shape loss."""

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()
        

    def forward(self, y_hat, y):
        """Calculate MSE loss.

        Args:
            y_hat (Tensor): Generated single tensor (B, 1, T).
            y (Tensor): Groundtruth single tensor (B, 1, T).

        Returns:
            Tensor: MSE loss value.

        """
        
        loss = self.loss(y_hat, y)*(y.shape[2]) * 48000* 10
        
        return loss
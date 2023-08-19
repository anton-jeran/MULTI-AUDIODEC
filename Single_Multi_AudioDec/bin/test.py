#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""Testing stage template."""

import os
import abc
import sys
import time
import yaml
import torch
import logging
import soundfile as sf
import cupy as cp
from cupyx.scipy.signal import fftconvolve
from tqdm import tqdm
from bin.utils import load_config

class TestGEN(abc.ABC):
    def __init__(
        self,
        args,
    ):
        # set logger
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )

        # device
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
            logging.info(f"device: cpu")
        else:
            self.device = torch.device('cuda')
            logging.info(f"device: gpu")
        
        # initialize attribute
        if hasattr(args, 'encoder'):
            self.encoder_checkpoint = args.encoder
            self.encoder_config = self._load_config(args.encoder)
        if hasattr(args, 'decoder'):
            self.decoder_checkpoint = args.decoder
            self.decoder_config = self._load_config(args.decoder)
        self.encoder = None
        self.decoder = None
        self.dataset = None
        self.outdir = None
    

    @abc.abstractmethod
    def initial_folder(self, output_name):
        pass
        

    @abc.abstractmethod    
    def load_dataset(self):
        pass
    
    
    @abc.abstractmethod
    def load_encoder(self):
        pass
    
    
    @abc.abstractmethod
    def load_decoder(self):
        pass
    
    
    @abc.abstractmethod
    def encode(self, x):
        pass
    
    
    @abc.abstractmethod
    def decode(self, z):
        pass

    
    def run(self):
        total_rtf = 0.0
        with torch.no_grad(), tqdm(self.dataset, desc="[test]") as pbar:
            for idx, (utt_id, x) in enumerate(pbar, 1):
                start = time.time()
                zq_speech, zq_rir = self.encode(x)
                y_speech,y_rir = self.decode(zq_speech, zq_rir)
                y_speech = y_speech.squeeze(0).transpose(1, 0).cpu().numpy() # T x C
                y_rir = y_rir.squeeze(0).transpose(1, 0).cpu().numpy() # T x C
                rtf = (time.time() - start) / (len(y_speech) / self.decoder_config['sampling_rate'])
                pbar.set_postfix({"RTF": rtf})
                total_rtf += rtf

                y_speech_cpu = cp.asarray(y_speech)
                y_rir_cpu = cp.asarray(y_rir)
                y_reverb_speech_ = cp.asnumpy(fftconvolve(y_speech_cpu,y_rir_cpu,axes=0)[0:y_speech.shape[0],:])
  


                clean_dir  = (os.path.join(self.outdir,"clean"))
                rir_dir  = (os.path.join(self.outdir,"rir"))
                reverb_dir  = (os.path.join(self.outdir,"reverb"))

                if (not os.path.exists(clean_dir)):

                    os.mkdir(clean_dir)
                    os.mkdir(rir_dir)
                    os.mkdir(reverb_dir)

                # output wav file
                self._save_wav(os.path.join(clean_dir, f"{utt_id}.wav"), y_speech)
                self._save_wav(os.path.join(rir_dir, f"{utt_id}.wav"), y_rir)
                self._save_wav(os.path.join(reverb_dir, f"{utt_id}.wav"), y_reverb_speech_)


        logging.info(
            "Finished generation of %d utterances (RTF = %.03f)." % (idx, (total_rtf / idx))
        )
    

    def _save_wav(self, file_name, audio):
        sf.write(
            file_name,
            audio,
            self.decoder_config['sampling_rate'],
            "PCM_16",
        )
    
    
    def _load_config(self, checkpoint, config_name='config.yml'):
        return load_config(checkpoint, config_name)

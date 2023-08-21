#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""PyTorch compatible dataset modules."""

import os
import soundfile as sf
from torch.utils.data import Dataset
from dataloader.utils import find_files
import pickle
import numpy as np

class SingleDataset(Dataset):
    def __init__(
        self,
        files,
        query="*.wav",
        load_fn=sf.read,
        return_utt_id=False,
        subset_num=-1,
    ):
        self.return_utt_id = return_utt_id
        self.load_fn = load_fn
        self.subset_num = subset_num
        self.files = files

        self.filenames = self._load_list(files[0], query)
        self.utt_ids = self._load_ids(self.filenames)
        self.dict_path_rir1 = os.path.join(files[0], "dictionary_ir1.pickle")
        self.dict_path_rir2 = os.path.join(files[0], "dictionary_ir2.pickle")
        self.dict_path_speech2 = os.path.join(files[0], "dictionary_speech2.pickle")
        

        if(len(files)>1):
            with open(self.dict_path_rir1, 'rb') as f_rir1:
                self.dictionary_rir1 = pickle.load(f_rir1)

            with open(self.dict_path_rir2, 'rb') as f_rir2:
                self.dictionary_rir2 = pickle.load(f_rir2)

            with open(self.dict_path_speech2, 'rb') as f_speech2:
                self.dictionary_speech2 = pickle.load(f_speech2)




    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        data = self._data(idx)
                
        if self.return_utt_id:
            items = utt_id, data
        else:
            items = data

        return items


    def __len__(self):
        return len(self.filenames)
    
    
    def _read_list(self, listfile):
        filenames = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                if len(line):
                    filenames.append(line)
        return filenames
    

    def _load_list(self, files, query):
        if isinstance(files, list):
            filenames = files
        else:
            if os.path.isdir(files):
                filenames = sorted(find_files(files, query,False))
            elif os.path.isfile(files):
                filenames = sorted(self._read_list(files))
            else:
                raise ValueError(f"{files} is not a list / existing folder or file!")
            
        if self.subset_num > 0:
            filenames = filenames[:self.subset_num]
        assert len(filenames) != 0, f"File list in empty!"
        return filenames
    
    
    def _load_ids(self, filenames):
        utt_ids = [
            os.path.splitext(os.path.basename(f))[0] for f in filenames
        ]
        return utt_ids
    

    def _data(self, idx):
        return self._load_data(self.filenames[idx], self.load_fn)
    

    def _load_data(self, filename, load_fn):
        data=[]

        reverb_path = os.path.join(self.files[0],filename)
        reverb_data, _ = load_fn(reverb_path, always_2d=True) # (T, C)
        reverb_data = reverb_data[0:96000,:]
        data = reverb_data

     
        if(len(self.files)>1):
            clean_path_1 = os.path.join(self.files[1],filename)
            clean_path_2 = os.path.join(self.files[1],self.dictionary_speech2[filename])
            rir_path_1 = os.path.join(self.files[2],self.dictionary_rir1[filename])
            rir_path_2 = os.path.join(self.files[2],self.dictionary_rir2[filename])
        
            clean_data_1, _ = load_fn(clean_path_1, always_2d=True) # (T, C)
            clean_data_2, _ = load_fn(clean_path_2, always_2d=True) # (T, C)


            rir_data_1, _ = load_fn(rir_path_1, always_2d=True) # (T, C)
            rir_data_2, _ = load_fn(rir_path_2, always_2d=True) # (T, C)

            rir_data = np.concatenate((rir_data_1,rir_data_2),axis=1)
        
            clean_data_1 =  clean_data_1[0:96000,:]
            clean_data_2 =  clean_data_2[0:96000,:]

            clean_data = np.concatenate((clean_data_1,clean_data_2),axis=1)



        # print("rir_data shape  ", rir_data.shape)
        # print("clean_data shape  ", clean_data.shape)
        # print("reverb_data shape  ", reverb_data.shape)


            data = [reverb_data, clean_data, rir_data]
        
        return data





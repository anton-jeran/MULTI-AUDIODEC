# MULTI-AUDIODEC: BINAURAL HIGH-FIDELITY NEURAL AUDIO CODEC FOR OVERLAPPED SPEECH

This is the official implementation of our binaural neural audio codec architecure of single-speaker speech signal and two-speaker overlapped speech signal. Our **MULTI-AUDIODEC** architecture is developed based on [**AUDIODEC**](https://github.com/facebookresearch/AudioDec) repository. Therefore our work is licensed under [**Creative Commons Attribution-NonCommercial 4.0 International License**](https://creativecommons.org/licenses/by-nc/4.0/). When you use this repository, please consider citing our work and  [**AUDIODEC**](https://github.com/facebookresearch/AudioDec).  

# Requirements

```
Python 3.8+
Cuda 11.0+
PyTorch 1.10+
numpy
pygsound
wavefile
tqdm
scipy
soundfile
librosa
cupy-cuda11x
torch_stoi
```

# Single Speaker Binaural Speech

**To train and test single speaker binaural speech go inside "Single_Multi_AudioDec/" folder.**

## Generating Binaural RIR

To generate 50,000 RIRs run the follwing code. To generate different number of RIRs, change variable **num_irs** (Line 47) in **sim_binaural_ir.py**. You can see generated Binaural RIRs under **binaural/** folder.

```
python3 sim_binaural_ir.py
```

## Augement Binaural Speech Dataset
Download VCTK or any clean speech dataset and divide into train,test and valid **e.g., corpus/train, corpus/test, corpus/valid**. Then make folder **output_speech** and run following command to augment binaural speech dataset

```
python3 augment_binaural_speech.py --speech corpus/train/ --ir binaural/ --out output_speech/train --nthreads 16
python3 augment_binaural_speech.py --speech corpus/valid/ --ir binaural/ --out output_speech/valid --nthreads 16
python3 augment_binaural_speech.py --speech corpus/test/ --ir binaural/ --out output_speech/test --nthreads 16
```

## Training our Multi_AudioDec with Metric Loss
We train our end-to-end network with only metric loss for 200,000 epoch. To train our network, run the following command 
```
bash submit_autoencoder.sh --stage 0 --tag_name "autoencoder/symAD_vctk_48000_hop300"
```

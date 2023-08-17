# MULTI-AUDIODEC: BINAURAL HIGH-FIDELITY NEURAL AUDIO CODEC FOR OVERLAPPED SPEECH

This is the official implementation of our binaural neural audio codec architecure of single-speaker speech signal and two-speaker overlapped speech signal. Our **MULTI-AUDIODEC** architecture is developed based on [**AUDIODEC**](https://github.com/facebookresearch/AudioDec) repository. Therefore our work is licensed under [**Creative Commons Attribution-NonCommercial 4.0 International License**](https://creativecommons.org/licenses/by-nc/4.0/). 

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
```

# Single Speaker Binaural Speech

**To train and test single speaker binaural speech go inside "Single_Multi_AudioDec/" folder.**

## Generating Binaural RIR

To generate 50,000 RIRs run the follwing code. To generate different number of RIRs, change variable **num_irs** (Line 47) in **sim_binaural_ir.py**. You can see generated Binaural RIRs under **binaural/** folder.

```
python3 sim_binaural_ir.py
```

## Augement Binaural Speech Dataset
Download VCTK or any clean speech dataset and divide into train,test and valid **e.g., corpus/train, corpus/test, corpus/valid**. Then make folder **output_speech**
#

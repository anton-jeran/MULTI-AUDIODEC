# M3-AUDIODEC: MULTI-CHANNEL MULTI-SPEAKER MULTI-SPATIAL AUDIO CODEC

This is the official implementation of our multi-channel multi-speaker multi-spatial neural audio codec architecture confgured for single-speaker speech signal and two-speaker overlapped speech signal. Our [**M3-AUDIODEC**](https://arxiv.org/abs/2309.07416]) architecture is developed based on [**AUDIODEC**](https://github.com/facebookresearch/AudioDec) repository. Therefore our work is licensed under [**Creative Commons Attribution-NonCommercial 4.0 International License**](https://creativecommons.org/licenses/by-nc/4.0/). When you use this repository, please consider citing our work [**M3-AUDIODEC**](https://arxiv.org/abs/2309.07416) and  [**AUDIODEC**](https://github.com/facebookresearch/AudioDec).  

The audio demos of M3-AUDIODEC are available [**here**](https://anton-jeran.github.io/MAD/). 

Our paper is available [**here**](https://arxiv.org/abs/2309.07416). 

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
sudo apt-get install p7zip-full  
```

# Single Speaker Binaural Speech

**To train and test single-speaker binaural speech go inside "Single_Multi_AudioDec/" folder.**

## Generating Binaural IR

To generate 50,000 BIRs run the following code. To generate different numbers of BIRs, change the variable **num_irs** (Line 47) in **sim_binaural_ir.py**. You can see generated Binaural RIRs under **binaural/** folder.

```
python3 sim_binaural_ir.py
```

## Augement Binaural Speech Dataset
Download VCTK or any clean speech dataset and divide into train,test and valid **e.g., corpus/train, corpus/test, corpus/valid**. To make clean speech of 2 seconds durations run the following command

```
python3 process_speech.py
```


Then make folder **output_speech** and run the following command to augment the binaural speech dataset

```
mkdir output_speech
python3 augment_binaural_speech.py --speech corpus/train/ --ir binaural/ --out output_speech/train --nthreads 16
python3 augment_binaural_speech.py --speech corpus/valid/ --ir binaural/ --out output_speech/valid --nthreads 16
python3 augment_binaural_speech.py --speech corpus/test/ --ir binaural/ --out output_speech/test --nthreads 16
```

## Trained Model and Test Data

To download our trained with encoder checkpoint at **200,000** and decoder checkpoint at **500,000** (**We tested on this model**). Run the following command
```
source download_model.sh
```
change the respective check point numbers to the variable **encoder_checkpoint**, **decoder_checkpoint** in **submit_codec_vctk.sh**.

To download our trained with encoder checkpoint at **500,000** and decoder checkpoint at **380,000**. Run the following command
```
source download_model_500.sh
```
change the respective check point numbers to the variable **encoder_checkpoint**, **decoder_checkpoint** in **submit_codec_vctk.sh**.

To download our test data, run the following command

```
source download_test_data.sh
```

## Training our M3-AUDIODEC with Metric Loss
We train our end-to-end network with only metric loss for 200,000 epochs. To train our network, run the following command 

```
bash submit_autoencoder.sh --start 0 --stop 0 --tag_name "autoencoder/symAD_vctk_48000_hop300"
```

We have configured it to run on 4 GPUs. To run on different numbers of GPUs change the **gpus:** parameter (Line-14) in **config/autoencoder/symAD_vctk_48000_hop300.yaml**
To run on different batch size, change **batch_size:** parameter (Line-193) in **config/autoencoder/symAD_vctk_48000_hop300.yaml**

To resume training on a saved model at a particular step (e.g., 200,000 steps) run the following command

```
bash submit_autoencoder.sh --start 1 --stop 1 --resumepoint 200000 --tag_name "autoencoder/symAD_vctk_48000_hop300"
```

After training for 200,000 steps, we freeze the encode, projector, and quantizer and we only train the decoder with adversarial loss. To replace the simple decoder with the HiFi-GAN vocoder, run the following command

```
bash submit_codec_vctk.sh --start 1 --stop 2
```

If you want to resume training on the saved HiFi-GAN vocoder at a particular step (e.g., 460,000) run the following command

```
bash submit_codec_vctk.sh --start 3 --resumepoint 460000
```

To test the trained model run the following command

```
bash submit_autoencoder.sh --start 2
```


# Two Speakers Overlapped Binaural Speech

**To train and test single-speaker binaural speech go inside "Single_Multi_AudioDec/" folder.**

## Generating Binaural IR

To generate 50,000 BIRs run the following code. To generate different numbers of BIRs, change the variable **num_irs** (Line 47) in **sim_binaural_ir.py**. You can see generated Binaural RIRs under **binaural/** folder.

```
python3 sim_binaural_ir.py
```

## Augement Binaural Speech Dataset
Download VCTK or any clean speech dataset and divide it into the train, test and valid **e.g., corpus/train, corpus/test, corpus/valid**. To make a clean speech of 2 seconds duration run the following command

```
python3 process_speech.py
```


Then make folder **output_speech** and run the following command to augment the binaural speech dataset

```
mkdir output_speech
python3 augment_overlap_binaural_speech.py --speech corpus/train/ --ir binaural/ --out output_speech/train --nthreads 16
python3 augment_overlap_binaural_speech.py --speech corpus/valid/ --ir binaural/ --out output_speech/valid --nthreads 16
python3 augment_overlap_binaural_speech.py --speech corpus/test/ --ir binaural/ --out output_speech/test --nthreads 16

```

## Trained Model and Test Data

To download our trained with encoder checkpoint at **200,000** and decoder checkpoint at **358,651**. Run the following command
```
source download_model.sh
```
change the respective check point numbers to the variable **encoder_checkpoint**, **decoder_checkpoint** in **submit_codec_vctk.sh**.


To download our test data, run the following command

```
source download_test_data.sh
```

## Training our M3-AUDIODEC with Metric Loss
We train our end-to-end network with only metric loss for 200,000 epochs. To train our network, run the following command 

```
bash submit_autoencoder.sh --start 0 --stop 0 --tag_name "autoencoder/symAD_vctk_48000_hop300"
```

We have configured it to run on 4 GPUs. To run on different numbers of GPUs change the **gpus:** parameter (Line-14) in **config/autoencoder/symAD_vctk_48000_hop300.yaml**
To run on different batch size, change **batch_size:** parameter (Line-193) in **config/autoencoder/symAD_vctk_48000_hop300.yaml**

To resume training on the saved model at a particular step (e.g., 200,000 steps) run the following command

```
bash submit_autoencoder.sh --start 1 --stop 1 --resumepoint 200000 --tag_name "autoencoder/symAD_vctk_48000_hop300"
```

To test the trained model run the following command

```
bash submit_autoencoder.sh --start 2
```

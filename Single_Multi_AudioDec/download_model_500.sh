gdown https://drive.google.com/uc?id=1Ln-Q8cvi_H_X7Xp06J4-lAsf0QSSBtww
7z e exp.zip 
mv checkpoint-500000steps.pkl exp/autoencoder/symAD_vctk_48000_hop300
mv *pkl exp/vocoder/AudioDec_v1_symAD_vctk_48000_hop300_clean
rm -rf AudioDec_v1_symAD_vctk_48000_hop300_clean vocoder autoencoder symAD_vctk_48000_hop300

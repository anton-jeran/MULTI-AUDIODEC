gdown https://drive.google.com/uc?id=1Ln-Q8cvi_H_X7Xp06J4-lAsf0QSSBtww
7z e exp.zip 
mv checkpoint-500000steps.pkl symAD_vctk_48000_hop300
mv symAD_vctk_48000_hop300 autoencoder
mv autoencoder exp
mv *pkl AudioDec_v1_symAD_vctk_48000_hop300_clean
mv AudioDec_v1_symAD_vctk_48000_hop300_clean vocoder
mv vocoder exp
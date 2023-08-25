import os
import soundfile as sf
import numpy as np

speech_path = "corpus/"

train_path = speech_path + "train/"

test_path =  speech_path + "test/"

valid_path =  speech_path + "valid/"

file_list_train = os.listdir(train_path)

for files in file_list_train:
	full_path = train_path + files

	speech,fs = sf.read(full_path)

	crop_length = 96000
	length = speech.shape[0]
	if(length<crop_length):
		zeros = np.zeros(crop_length-length)
		speech = np.concatenate([speech,zeros])
	else:
		speech = speech[0:96000]

	

	sf.write(full_path,speech,fs)


file_list_test = os.listdir(test_path)

for files in file_list_test:
	full_path = test_path + files

	speech,fs = sf.read(full_path)

	crop_length = 96000
	length = speech.shape[0]
	if(length<crop_length):
		zeros = np.zeros(crop_length-length)
		speech = np.concatenate([speech,zeros])
	else:
		speech = speech[0:96000]

	

	sf.write(full_path,speech,fs)

file_list_valid = os.listdir(valid_path)

for files in file_list_valid:
	full_path = valid_path + files

	speech,fs = sf.read(full_path)

	crop_length = 96000
	length = speech.shape[0]
	if(length<crop_length):
		zeros = np.zeros(crop_length-length)
		speech = np.concatenate([speech,zeros])
	else:
		speech = speech[0:96000]

	

	sf.write(full_path,speech,fs)
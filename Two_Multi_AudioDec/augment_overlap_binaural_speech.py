import os
import numpy as np
import argparse
from multiprocessing import Pool
import random
import soundfile as sf
import scipy.signal as ssi
from tqdm import tqdm
import utility
import pickle

def augment_data(speech_path1, output_path, irfile_path1, irfile_path2,speech_path2):


    speech1, fs_s1 = sf.read(speech_path1)
    speech2, fs_s2 = sf.read(speech_path2)

    speech_length1 = speech1.shape[0]
    speech_length2 = speech2.shape[0]  

    if(speech_length1>96000):
        speech1 = speech1[0:96000]
            
    else:
        zeros_len1 = 96000 - speech_length1
        zeros_lis1 = np.zeros(zeros_len1)
        speech1 = np.concatenate([speech1,zeros_lis1])
   

    if(speech_length2>96000):
        speech2 = speech2[0:96000]
            
    else:
        zeros_len2 = 96000 - speech_length2
        zeros_lis2 = np.zeros(zeros_len2)
        speech2 = np.concatenate([speech2,zeros_lis2])

    if np.issubdtype(speech1.dtype, np.integer):
        speech1 = utility.pcm2float(speech1, 'float32')   

    if np.issubdtype(speech2.dtype, np.integer):
        speech2 = utility.pcm2float(speech2, 'float32')



    # convolution
    if irfile_path1:
        IR1, fs_i1 = sf.read(irfile_path1)
        IR2, fs_i2 = sf.read(irfile_path2)

        IR_length1 = IR1.shape[0]
        IR_length2 = IR2.shape[0]
    
        if(IR_length1>fs_s1):
            IR1 = IR1[0:fs_s1,:]
            
        else:
            zeros_len1 = fs_s1 - IR_length1
            zeros_lis1 = np.zeros([zeros_len1,2])
            IR1 = np.concatenate([IR1,zeros_lis1])

        if np.issubdtype(IR1.dtype, np.integer):
            IR1 = utility.pcm2float(IR1, 'float32')

        if(IR_length2>fs_s2):
            IR2 = IR2[0:fs_s2,:]
            
        else:
            zeros_len2 = fs_s2 - IR_length2
            zeros_lis2 = np.zeros([zeros_len2,2])
            IR2 = np.concatenate([IR2,zeros_lis2])

        if np.issubdtype(IR2.dtype, np.integer):
            IR2 = utility.pcm2float(IR2, 'float32')



        # speech = utility.convert_samplerate(speech, fs_s, fs_i)
        # fs_s = fs_i
        # eliminate delays due to direct path propagation
        # direct_idx = np.argmax(np.fabs(IR))
        #print('speech {} direct index is {} of total {} samples'.format(speech_path, direct_idx, len(IR)))

        # temp = utility.smart_convolve(speech, IR[direct_idx:])
        temp10 = utility.smart_convolve(speech1, IR1[:,0])
        temp11 = utility.smart_convolve(speech1, IR1[:,1])

        temp20 = utility.smart_convolve(speech2, IR2[:,0])
        temp21 = utility.smart_convolve(speech2, IR2[:,1])

        temp0 = temp10 + temp20
        temp1 = temp11 + temp21

        temp =np.transpose(np.concatenate(([temp0], [temp1]),axis=0))
        
        speech = np.array(temp)
    # # adding noises
    # if noise_path:
    #     noise, fs_n = sf.read(noise_path)
    #     if len(noise.shape) != 1:
    #         print("noise file should be single channel")
    #         return -1
    #     if np.issubdtype(noise.dtype, np.integer):
    #         noise = utility.pcm2float(noise, 'float32')
    #     noise = utility.convert_samplerate(noise, fs_n, fs_s)
    #     fs_n = fs_s       
    #     speech_len = len(speech)
    #     noise_len = len(noise)
    #     nrep = int(speech_len * 2 / noise_len)
    #     if nrep >= 1:
    #         noise = np.repeat(noise, nrep + 1)
    #         noise_len = len(noise)
    #     start = np.random.randint(noise_len - speech_len)
    #     noise = noise[start:(start + speech_len)]

    #     signal_power = utility.calc_valid_power(speech)
    #     noise_power = utility.calc_valid_power(noise)
    #     K = (signal_power / noise_power) * np.power(10, -SNR / 10)

    #     new_noise = np.sqrt(K) * noise
    #     speech = speech + new_noise
    maxval = np.max(np.fabs(speech))
    if maxval == 0:
        print("file {} not saved due to zero strength".format(speech_path))
        return -1
    if maxval >= 1:
        amp_ratio = 0.99 / maxval
        speech = speech * amp_ratio
    sf.write(output_path, speech, fs_s1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='augment',
                                     description="""Script to augment dataset""")
    parser.add_argument("--ir", "-i", default=None, help="Directory of IR files", type=str)
    # parser.add_argument("--noise", "-no", default=None, help="Directory of noise files", type=str)
    parser.add_argument("--speech", "-sp", required=True, help="Directory of speech files", type=str)
    parser.add_argument("--out", "-o", required=True, help="Output folder path", type=str)
    parser.add_argument("--seed", "-s", default=0, help="Random seed", type=int)
    parser.add_argument("--nthreads", "-n", type=int, default=32, help="Number of threads to use")

    args = parser.parse_args()
    speech_folder = args.speech
    # noise_folder = args.noise
    ir_folder = args.ir
    output_folder = args.out
    nthreads = args.nthreads

    embedding_list_ir1={}
    embedding_list_ir2={}
    embedding_list_speech2={}
    
    # force input and output folder to have the same ending format (i.e., w/ or w/o slash)
    speech_folder = os.path.join(speech_folder, '')
    output_folder = os.path.join(output_folder, '')    

    add_reverb = True if ir_folder else False
    # add_noise = True if noise_folder else False

    assert os.path.exists(speech_folder)
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)
    # if add_noise:
    #     assert os.path.exists(noise_folder)
    if add_reverb:
        assert os.path.exists(ir_folder)

    speechlist = [os.path.join(root, name) for root, dirs, files in os.walk(speech_folder)
              for name in files if name.endswith(".wav")]
    irlist = [os.path.join(root, name) for root, dirs, files in os.walk(ir_folder)
              for name in files if name.endswith(".wav")] if add_reverb else []
    # noiselist = [os.path.join(root, name) for root, dirs, files in os.walk(noise_folder)
    #           for name in files if name.endswith(".wav")] if add_noise else []

    # apply_async callback
    

    pbar = tqdm(total=len(speechlist))
    def update(*a):
        pbar.update()
    try:
        # # Create a pool to communicate with the worker threads
        pool = Pool(processes=nthreads)
        for speech_path1 in speechlist:
            ir_sample1 = random.choice(irlist) 
            ir_sample2 = random.choice(irlist) 
            speech_path2 = random.choice(speechlist)
            # noise_sample = random.choice(noiselist) if add_noise else None
            # SNR = np.random.uniform(10, 20)
            output_path = speech_path1.replace(speech_folder, output_folder)
            embedding_list_ir1[output_path.split("/")[-1]] =  ir_sample1.split("/")[-1]
            embedding_list_ir2[output_path.split("/")[-1]] =  ir_sample2.split("/")[-1]
            embedding_list_speech2[output_path.split("/")[-1]] =  speech_path2.split("/")[-1]
            # embeddings = [speech_path,output_path,ir_sample]
            # embedding_list.append(embeddings)
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            pool.apply_async(augment_data, args=(speech_path1, output_path, ir_sample1,ir_sample2,speech_path2), callback=update)
    except Exception as e:
        print(str(e))
        pool.close()
    pool.close()
    pool.join()

    embeddings_pickle_ir1 =output_folder+"dictionary_ir1.pickle"
    with open(embeddings_pickle_ir1, 'wb') as f:
        pickle.dump(embedding_list_ir1, f, protocol=2)

    embeddings_pickle_ir2 =output_folder+"dictionary_ir2.pickle"
    with open(embeddings_pickle_ir2, 'wb') as f:
        pickle.dump(embedding_list_ir2, f, protocol=2)

    embeddings_pickle_speech2 =output_folder+"dictionary_speech2.pickle"
    with open(embeddings_pickle_speech2, 'wb') as f:
        pickle.dump(embedding_list_speech2, f, protocol=2)


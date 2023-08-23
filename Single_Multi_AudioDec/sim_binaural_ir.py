import numpy as np
import pygsound as ps
import os
from wavefile import WaveWriter, Format
import random
from multiprocessing import Pool
from tqdm import tqdm
import utility

def generate_binaural(src_coord,lis_coord,room_coord):
    # Simulation using .obj file (and an optional .mtl file)
    ctx = ps.Context()
    ctx.diffuse_count = 20000
    ctx.specular_count = 2000
    # ctx.specular_depth = 10
    ctx.sample_rate=48000
    ctx.channel_type = ps.ChannelLayoutType.binaural
    ctx.normalize = False


    file_name = "binaural/r-"+str(round(room_coord[0],1)) +"_"+str(round(room_coord[1],1)) +"_"+str(round(room_coord[2],1)) +"_"+"l-"+str(round(lis_coord[0],1)) +"_"+str(round(lis_coord[1],1)) +"_"+str(round(lis_coord[2],1)) +"_"+"s-"+str(round(src_coord[0],1)) +"_"+str(round(src_coord[1],1)) +"_"+str(round(src_coord[2],1)) +".wav"
    # src_coord = [1, 1, 0.5]
    # lis_coord = [5, 3, 0.5]

    src = ps.Source(src_coord)
    src.radius = 0.01
    src.power = 1.0

    lis = ps.Listener(lis_coord)
    lis.radius = 0.01

    
    # Simulation using a shoebox definition
    mesh = ps.createbox(room_coord[0], room_coord[1], room_coord[2], 0.5, 0.1)
    scene = ps.Scene()
    scene.setMesh(mesh)

    res = scene.computeIR([src_coord], [lis_coord], ctx)    # use default source and listener settings if you only pass coordinates
    audio_data = np.array(res['samples'][0][0])
    
    if(audio_data.shape[1]>48000):
        audio_data = audio_data[:,0:48000]
    else:
        zeros = np.zeros([2,48000-audio_data.shape[1]])
        audio_data = np.concatenate((audio_data,zeros),axis=1)
    # print("audio_data shape ",audio_data.shape)


    with WaveWriter(file_name, channels=audio_data.shape[0], samplerate=int(res['rate'])) as w2:
        w2.write(audio_data)


if __name__ == '__main__':

    nthreads = 16
    num_irs =50000
    pbar = tqdm(total=num_irs)
    if(not os.path.exists("binaural/")):
        os.mkdir("binaural/")
        
    def update(*a):
        pbar.update()
    try:
        # # Create a pool to communicate with the worker threads
        pool = Pool(processes=nthreads)
        for i in range(num_irs):

            room_x = (random.randrange(0,1000)/1000) * 12 + 2
            room_y = (random.randrange(0,1000)/1000) * 12 + 2
            room_z = (random.randrange(0,1000)/1000) * 3 + 2

            lis_coord_x = ((random.randrange(0,1000)/1000) * (room_x-0.1)) #- (room_x/2)
            lis_coord_y = ((random.randrange(0,1000)/1000) * (room_y-0.1)) #- (room_y/2)
            lis_coord_z = ((random.randrange(0,1000)/1000) * (room_z-0.1)) #- (room_z/2)

            src_coord_x = ((random.randrange(0,1000)/1000) * (room_x-0.1)) #- (room_x/2)
            src_coord_y = ((random.randrange(0,1000)/1000) * (room_y-0.1)) #- (room_y/2)
            src_coord_z = ((random.randrange(0,1000)/1000) * (room_z-0.1)) #- (room_z/2)


            src_coord = [lis_coord_x, lis_coord_y, lis_coord_z]
            lis_coord = [src_coord_x, src_coord_y, src_coord_z]

            room_coord = [room_x,room_y,room_z]

            generate_binaural(src_coord,lis_coord,room_coord)

    except Exception as e:
        print(str(e))
        pool.close()
    pool.close()
    pool.join()

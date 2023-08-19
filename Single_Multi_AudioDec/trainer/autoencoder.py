#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""Training flow of symmetric codec."""

import logging
import torch
from trainer.trainerGAN import TrainerVQGAN

import numpy as np
from wavefile import WaveWriter, Format
import os
import shutil
import torch.nn as nn
import torch.nn.functional as F
import cupy as cp
from cupyx.scipy.signal import fftconvolve


class Trainer(TrainerVQGAN):
    def __init__(
        self,
        steps,
        epochs,
        filters,
        data_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
    ):
        super(Trainer, self).__init__(
           steps=steps,
           epochs=epochs,
           filters=filters,
           data_loader=data_loader,
           model=model,
           criterion=criterion,
           optimizer=optimizer,
           scheduler=scheduler,
           config=config,
           device=device,
        )
        self.fix_encoder = False
        self.paradigm = config.get('paradigm', 'efficient') 
        self.generator_start = config.get('start_steps', {}).get('generator', 0)
        self.discriminator_start = config.get('start_steps', {}).get('discriminator', 200000)
        self.filters = filters


    def _train_step(self, batch):
        """Single step of training."""
        mode = 'train'
        rs,cs,rir = batch
        
        disc_length = 9600
        disc_num = self.steps %(rs.shape[2]//disc_length)

        # print("disc_num ",disc_num)

        rs = rs.to(self.device)
        cs = cs.to(self.device)
        rir = rir.to(self.device)

        # check generator step
        if self.steps < self.generator_start:
            self.generator_train = False
        else:
            self.generator_train = True
            
        # check discriminator step
        if self.steps < self.discriminator_start:
            self.discriminator_train = False
        else:
            self.discriminator_train = True
            if (not self.fix_encoder) and (self.paradigm == 'efficient'):
                # fix encoder, quantizer, and codebook
                for parameter in self.model["generator"].encoder.parameters():
                    parameter.requires_grad = False
                for parameter in self.model["generator"].projector_speech.parameters():
                    parameter.requires_grad = False
                for parameter in self.model["generator"].quantizer_speech.parameters():
                    parameter.requires_grad = False
                for parameter in self.model["generator"].projector_rir.parameters():
                    parameter.requires_grad = False
                for parameter in self.model["generator"].quantizer_rir.parameters():
                    parameter.requires_grad = False
                self.fix_encoder = True
                logging.info("Encoder, projector, quantizer, and codebook are fixed")
        
        # check codebook updating
        if self.fix_encoder:
            self.model["generator"].quantizer_speech.codebook.eval()
            self.model["generator"].quantizer_rir.codebook.eval()

        #######################
        #      Generator      #
        #######################
        if self.generator_train:
            # initialize generator loss
            gen_loss = 0.0

            # main genertor operation
            y_speech_, y_rir_, zq_speech, zq_rir, z_speech, z_rir, vqloss_speech, vqloss_rir, perplexity_speech, perplexity_rir,  = nn.parallel.data_parallel(self.model["generator"],rs,self.gpus)
            
            y_speech_cpu = cp.asarray(y_speech_.to("cpu").detach())
            y_rir_cpu = cp.asarray(y_rir_.to("cpu").detach())
            y_reverb_speech_ = torch.tensor(fftconvolve(y_speech_cpu,y_rir_cpu,axes=2)[:,:,0:y_speech_.shape[2]],device='cuda')

            

            # perplexity info
            self._perplexity(perplexity_speech, label="speech", mode=mode)
            self._perplexity(perplexity_rir, label="rir", mode=mode)

            # vq loss
            gen_loss += self._vq_loss(vqloss_speech, label="speech", mode=mode)
            gen_loss += self._vq_loss(vqloss_rir, label="rir", mode=mode)
            
            # metric loss
            gen_loss += self._metric_loss_speech(y_speech_, cs, mode=mode)
            gen_loss += self._metric_loss_reverb_speech(y_reverb_speech_, rs, mode=mode)
            gen_loss += self._metric_loss_rir(y_rir_, rir, self.filters,mode=mode)
            
            # adversarial loss
            if self.discriminator_train:
               
                p_speech_ = nn.parallel.data_parallel(self.model["discriminator_speech"],y_speech_[:,:,(disc_num*disc_length):((disc_num+1)*disc_length)],self.gpus)
                if self.config["use_feat_match_loss"]:
                    with torch.no_grad():
                        p_speech = nn.parallel.data_parallel(self.model["discriminator_speech"],cs[:,:,(disc_num*disc_length):((disc_num+1)*disc_length)],self.gpus) 
                else:
                    p_speech = None
                gen_loss += self._adv_loss(p_speech_, p_speech, mode=mode)


                y_reverb_speech_0 = y_reverb_speech_[:,0,:].reshape([y_reverb_speech_.shape[0],1,y_reverb_speech_.shape[2]])
                y_reverb_speech_1 = y_reverb_speech_[:,1,:].reshape([y_reverb_speech_.shape[0],1,y_reverb_speech_.shape[2]])

                rs0 = rs[:,0,:].reshape([rs.shape[0],1,rs.shape[2]])
                rs1 = rs[:,1,:].reshape([rs.shape[0],1,rs.shape[2]])
                
                
                p_reverb_0 = nn.parallel.data_parallel(self.model["discriminator_reverb"],y_reverb_speech_0[:,:,(disc_num*disc_length):((disc_num+1)*disc_length)],self.gpus) 
                if self.config["use_feat_match_loss"]:
                    with torch.no_grad():
                        p_reverb0 = nn.parallel.data_parallel(self.model["discriminator_reverb"],rs0[:,:,(disc_num*disc_length):((disc_num+1)*disc_length)],self.gpus) 
    
                else:
                    p_reverb0 = None
                gen_loss += self._adv_loss(p_reverb_0, p_reverb0, mode=mode)
    
                p_reverb_1 = nn.parallel.data_parallel(self.model["discriminator_reverb"],y_reverb_speech_1[:,:,(disc_num*disc_length):((disc_num+1)*disc_length)],self.gpus) 
                if self.config["use_feat_match_loss"]:
                    with torch.no_grad():
                        p_reverb1 = nn.parallel.data_parallel(self.model["discriminator_reverb"],rs1[:,:,(disc_num*disc_length):((disc_num+1)*disc_length)],self.gpus) 
    
                else:
                    p_reverb1 = None
                gen_loss += self._adv_loss(p_reverb_1, p_reverb1, mode=mode)

            # update generator
            self._record_loss('generator_loss', gen_loss, mode=mode)
            self._update_generator(gen_loss)

        #######################
        #    Discriminator    #
        #######################
        if self.discriminator_train:
            # re-compute y_ which leads better quality
            with torch.no_grad():
                y_speech_,y_rir_, _, _, _, _, _, _, _, _ = nn.parallel.data_parallel(self.model["generator"],rs,self.gpus) 

                y_speech_cpu = cp.asarray(y_speech_.to("cpu").detach())
                y_rir_cpu = cp.asarray(y_rir_.to("cpu").detach())
                y_reverb_speech_ = torch.tensor(fftconvolve(y_speech_cpu,y_rir_cpu,axes=2)[:,:,0:y_speech_.shape[2]],device='cuda')
            
            p_speech = nn.parallel.data_parallel(self.model["discriminator_speech"],cs[:,:,(disc_num*disc_length):((disc_num+1)*disc_length)],self.gpus) 
            p_speech_ = nn.parallel.data_parallel(self.model["discriminator_speech"],y_speech_[:,:,(disc_num*disc_length):((disc_num+1)*disc_length)].detach(),self.gpus) 
            dis_loss_speech = self._dis_loss_speech(p_speech_, p_speech, mode=mode)
            y_reverb_speech_0 = y_reverb_speech_[:,0,:].reshape([y_reverb_speech_.shape[0],1,y_reverb_speech_.shape[2]])
            y_reverb_speech_1 = y_reverb_speech_[:,1,:].reshape([y_reverb_speech_.shape[0],1,y_reverb_speech_.shape[2]])

            rs0 = rs[:,0,:].reshape([rs.shape[0],1,rs.shape[2]])
            rs1 = rs[:,1,:].reshape([rs.shape[0],1,rs.shape[2]])

            dis_loss_reverb = 0

            

            p_reverb0 = nn.parallel.data_parallel(self.model["discriminator_reverb"],rs0[:,:,(disc_num*disc_length):((disc_num+1)*disc_length)],self.gpus)  
            p_reverb_0 = nn.parallel.data_parallel(self.model["discriminator_reverb"],y_reverb_speech_0[:,:,(disc_num*disc_length):((disc_num+1)*disc_length)].detach(),self.gpus)
            dis_loss_reverb = dis_loss_reverb + self._dis_loss_reverb(p_reverb_0, p_reverb0, mode=mode)
                
            p_reverb1 = nn.parallel.data_parallel(self.model["discriminator_reverb"],rs1[:,:,(disc_num*disc_length):((disc_num+1)*disc_length)],self.gpus)  
            p_reverb_1 = nn.parallel.data_parallel(self.model["discriminator_reverb"],y_reverb_speech_1[:,:,(disc_num*disc_length):((disc_num+1)*disc_length)].detach(),self.gpus) 
            dis_loss_reverb = dis_loss_reverb + self._dis_loss_reverb(p_reverb_1, p_reverb1, mode=mode)
            # discriminator loss & update discriminator
            # self._update_discriminator(self._dis_loss_speech(p_speech_, p_speech, mode=mode), self._dis_loss_reverb(p_reverb_0, p_reverb0, mode=mode), self._dis_loss_reverb(p_reverb_1, p_reverb1, mode=mode))
            self._update_discriminator(dis_loss_speech, dis_loss_reverb)

        if(self.steps%5000==0):

            speech_path = os.path.join(self.save_path,"Speech")
            rir_path = os.path.join(self.save_path,"RIR")


            if(not os.path.exists(self.save_path)):
                os.mkdir(self.save_path)
                os.mkdir(speech_path)
                os.mkdir(rir_path)
            
            step_num = "step"+str(self.steps)

            speech_step_path = os.path.join(speech_path ,step_num)
            rir_step_path = os.path.join(rir_path ,step_num)

            speech_step_path_real = os.path.join(speech_step_path ,"real_sample/")
            speech_step_path_fake = os.path.join(speech_step_path ,"fake_sample/")
            speech_step_path_input = os.path.join(speech_step_path ,"input_sample/")
            speech_step_path_reverb = os.path.join(speech_step_path ,"reverb_sample/")

            rir_step_path_real = os.path.join(rir_step_path ,"real_sample/")
            rir_step_path_fake = os.path.join(rir_step_path ,"fake_sample/")

            # print("came here ")
            if(os.path.exists(speech_step_path)):
                shutil.rmtree(speech_step_path)
            os.mkdir(speech_step_path)
            os.mkdir(speech_step_path_real)
            os.mkdir(speech_step_path_fake)
            os.mkdir(speech_step_path_input)
            os.mkdir(speech_step_path_reverb)


            if(os.path.exists(rir_step_path)):
                shutil.rmtree(rir_step_path)
            os.mkdir(rir_step_path)
            os.mkdir(rir_step_path_real)
            os.mkdir(rir_step_path_fake)




            for i in range(rir.shape[0]):
                
                real_RIR_path = rir_step_path_real +str(i)+".wav" 
                fake_RIR_path = rir_step_path_fake+str(i)+".wav"
                fs =48000
                real_IR = np.array(rir[i].to("cpu").detach())
                generated_IR = np.array(y_rir_[i].to("cpu").detach())

                r = WaveWriter(real_RIR_path, channels=2, samplerate=fs)
                r.write(np.array(real_IR))
                f = WaveWriter(fake_RIR_path, channels=2, samplerate=fs)
                f.write(np.array(generated_IR))

            for i in range(cs.shape[0]):
                
                real_SPEECH_path = speech_step_path_real +str(i)+".wav" 
                fake_SPEECH_path = speech_step_path_fake+str(i)+".wav"
                input_SPEECH_path = speech_step_path_input+str(i)+".wav"
                reverb_SPEECH_path = speech_step_path_reverb+str(i)+".wav"


                fs =48000
                real_SPEECH = np.array(cs[i].to("cpu").detach())
                generated_SPEECH= np.array(y_speech_[i].to("cpu").detach())
                input_SPEECH= np.array(rs[i].to("cpu").detach())
                reverb_SPEECH= np.array(y_reverb_speech_[i].to("cpu").detach())

                r = WaveWriter(real_SPEECH_path, channels=1, samplerate=fs)
                r.write(np.array(real_SPEECH))
                f = WaveWriter(fake_SPEECH_path, channels=1, samplerate=fs)
                f.write(np.array(generated_SPEECH))

                i = WaveWriter(input_SPEECH_path, channels=2, samplerate=fs)
                i.write(np.array(input_SPEECH))
                re = WaveWriter(reverb_SPEECH_path, channels=2, samplerate=fs)
                re.write(np.array(reverb_SPEECH))

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()


    @torch.no_grad()
    def _eval_step(self, batch,steps):
        """Single step of evaluation."""
        mode = 'eval'
        rs,cs,rir = batch
        rs = rs.to(self.device)
        cs = cs.to(self.device)
        rir = rir.to(self.device)

        disc_length = 9600
        disc_num = steps %(rs.shape[2]//disc_length)
        
        # initialize generator loss
        gen_loss = 0.0

        # main genertor operation
        y_speech_, y_rir_, zq_speech, zq_rir, z_speech, z_rir, vqloss_speech, vqloss_rir, perplexity_speech, perplexity_rir = nn.parallel.data_parallel(self.model["generator"],rs,self.gpus)

        y_speech_cpu = cp.asarray(y_speech_.to("cpu").detach())
        y_rir_cpu = cp.asarray(y_rir_.to("cpu").detach())
        y_reverb_speech_ = torch.tensor(fftconvolve(y_speech_cpu,y_rir_cpu,axes=2)[:,:,0:y_speech_.shape[2]],device='cuda')

        

        # perplexity info
        self._perplexity(perplexity_speech, label="speech", mode=mode)
        self._perplexity(perplexity_rir, label="rir", mode=mode)


        # vq_loss
        gen_loss += self._vq_loss(vqloss_speech, label="speech", mode=mode)
        gen_loss += self._vq_loss(vqloss_rir, label="rir", mode=mode)
        
        # metric loss
        gen_loss += self._metric_loss_speech(y_speech_, cs, mode=mode)
        gen_loss += self._metric_loss_reverb_speech(y_reverb_speech_, rs, mode=mode)
        gen_loss += self._metric_loss_rir(y_rir_, rir,self.filters, mode=mode)

        if self.discriminator_train:
            # adversarial loss           
            
            p_speech = nn.parallel.data_parallel(self.model["discriminator_speech"],cs[:,:,(disc_num*disc_length):((disc_num+1)*disc_length)],self.gpus) 
            p_speech_ = nn.parallel.data_parallel(self.model["discriminator_speech"],y_speech_[:,:,(disc_num*disc_length):((disc_num+1)*disc_length)].detach(),self.gpus) 
            gen_loss += self._adv_loss(p_speech_, p_speech, mode=mode)
            
            y_reverb_speech_0 = y_reverb_speech_[:,0,:].reshape([y_reverb_speech_.shape[0],1,y_reverb_speech_.shape[2]])
            y_reverb_speech_1 = y_reverb_speech_[:,1,:].reshape([y_reverb_speech_.shape[0],1,y_reverb_speech_.shape[2]])

            rs0 = rs[:,0,:].reshape([rs.shape[0],1,rs.shape[2]])
            rs1 = rs[:,1,:].reshape([rs.shape[0],1,rs.shape[2]])

            p_reverb0 = nn.parallel.data_parallel(self.model["discriminator_reverb"],rs0[:,:,(disc_num*disc_length):((disc_num+1)*disc_length)],self.gpus) 
            p_reverb_0 = nn.parallel.data_parallel(self.model["discriminator_reverb"],y_reverb_speech_0[:,:,(disc_num*disc_length):((disc_num+1)*disc_length)].detach(),self.gpus) 
            gen_loss += self._adv_loss(p_reverb_0, p_reverb0, mode=mode)

            p_reverb1 = nn.parallel.data_parallel(self.model["discriminator_reverb"],rs1[:,:,(disc_num*disc_length):((disc_num+1)*disc_length)],self.gpus) 
            p_reverb_1 = nn.parallel.data_parallel(self.model["discriminator_reverb"],y_reverb_speech_1[:,:,(disc_num*disc_length):((disc_num+1)*disc_length)].detach(),self.gpus) 
            gen_loss += self._adv_loss(p_reverb_1, p_reverb0, mode=mode)

            # discriminator loss

            self._dis_loss_speech(p_speech_, p_speech, mode=mode)
            self._dis_loss_reverb(p_reverb_0, p_reverb0, mode=mode)
            self._dis_loss_reverb(p_reverb_1, p_reverb1, mode=mode)


           

        # generator loss
        self._record_loss('generator_loss', gen_loss, mode=mode)


        speech_path = os.path.join(self.save_path,"Speech_Eval")
        rir_path = os.path.join(self.save_path,"RIR_Eval")


        if(not os.path.exists(speech_path)):
            # os.mkdir(self.save_path)
            os.mkdir(speech_path)
            os.mkdir(rir_path)
         
        step_num = "step"+str(steps)

        speech_step_path = os.path.join(speech_path ,step_num)
        rir_step_path = os.path.join(rir_path ,step_num)

        speech_step_path_real = os.path.join(speech_step_path ,"real_sample/")
        speech_step_path_fake = os.path.join(speech_step_path ,"fake_sample/")
        speech_step_path_input = os.path.join(speech_step_path ,"input_sample/")
        speech_step_path_reverb = os.path.join(speech_step_path ,"reverb_sample/")

        rir_step_path_real = os.path.join(rir_step_path ,"real_sample/")
        rir_step_path_fake = os.path.join(rir_step_path ,"fake_sample/")

        # print("came here ")
        if(os.path.exists(speech_step_path)):
            shutil.rmtree(speech_step_path)
        os.mkdir(speech_step_path)
        os.mkdir(speech_step_path_real)
        os.mkdir(speech_step_path_fake)
        os.mkdir(speech_step_path_input)
        os.mkdir(speech_step_path_reverb)


        if(os.path.exists(rir_step_path)):
            shutil.rmtree(rir_step_path)
        os.mkdir(rir_step_path)
        os.mkdir(rir_step_path_real)
        os.mkdir(rir_step_path_fake)




        for i in range(rir.shape[0]):
            
            real_RIR_path = rir_step_path_real +str(i)+".wav" 
            fake_RIR_path = rir_step_path_fake+str(i)+".wav"
            fs =48000
            real_IR = np.array(rir[i].to("cpu").detach())
            generated_IR = np.array(y_rir_[i].to("cpu").detach())

            r = WaveWriter(real_RIR_path, channels=2, samplerate=fs)
            r.write(np.array(real_IR))
            f = WaveWriter(fake_RIR_path, channels=2, samplerate=fs)
            f.write(np.array(generated_IR))

        for i in range(cs.shape[0]):
            
            real_SPEECH_path = speech_step_path_real +str(i)+".wav" 
            fake_SPEECH_path = speech_step_path_fake+str(i)+".wav"
            input_SPEECH_path = speech_step_path_input+str(i)+".wav"
            reverb_SPEECH_path = speech_step_path_reverb+str(i)+".wav"


            fs =48000
            real_SPEECH = np.array(cs[i].to("cpu").detach())
            generated_SPEECH= np.array(y_speech_[i].to("cpu").detach())
            input_SPEECH= np.array(rs[i].to("cpu").detach())
            reverb_SPEECH= np.array(y_reverb_speech_[i].to("cpu").detach())

            r = WaveWriter(real_SPEECH_path, channels=1, samplerate=fs)
            r.write(np.array(real_SPEECH))
            f = WaveWriter(fake_SPEECH_path, channels=1, samplerate=fs)
            f.write(np.array(generated_SPEECH))

            i = WaveWriter(input_SPEECH_path, channels=2, samplerate=fs)
            i.write(np.array(input_SPEECH))
            re = WaveWriter(reverb_SPEECH_path, channels=2, samplerate=fs)
            re.write(np.array(reverb_SPEECH))

     

        

       


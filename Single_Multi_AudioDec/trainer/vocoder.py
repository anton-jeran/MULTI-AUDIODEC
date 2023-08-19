#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""Training flow of GAN-based vocoder."""

import logging
import torch
from trainer.trainerGAN import TrainerGAN
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from wavefile import WaveWriter, Format
import os
import shutil
import cupy as cp
from cupyx.scipy.signal import fftconvolve

class Trainer(TrainerGAN):
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
        self.fix_analyzer = False
        self.generator_start = config.get("generator_train_start_steps", 0)
        self.discriminator_start = config.get("discriminator_train_start_steps", 0)


    def _train_step(self, batch):
        """Train model one step."""
        mode = 'train'
        rs,cs,rir = batch
        
        disc_length = 9600
        disc_num = self.steps %(rs.shape[2]//disc_length)

        rs = rs.to(self.device)
        cs = cs.to(self.device)
        rir = rir.to(self.device)

        # fix analyzer
        if not self.fix_analyzer:    
            for parameter in self.model["analyzer"].parameters():
                parameter.requires_grad = False
            self.fix_analyzer = True
            logging.info("Analyzer is fixed!")
        self.model["analyzer"].eval()

        #######################
        #      Generator      #
        #######################
        if self.steps > self.generator_start:
            # initialize generator loss
            gen_loss = 0.0

            # main genertor operation
            e_speech, e_rir = self.model["analyzer"].encoder(rs)
           

            z_speech = self.model["analyzer"].projector_speech(e_speech)
            zq_speech, _, _ = self.model["analyzer"].quantizer_speech(z_speech)

            z_rir = self.model["analyzer"].projector_rir(e_rir)
            zq_rir, _, _ = self.model["analyzer"].quantizer_rir(z_rir)
            

            y_speech_, y_rir_ = nn.parallel.data_parallel(self.model["generator"],(zq_speech,zq_rir),self.gpus)

            y_speech_cpu = cp.asarray(y_speech_.to("cpu").detach())
            y_rir_cpu = cp.asarray(y_rir_.to("cpu").detach())
            y_reverb_speech_ = torch.tensor(fftconvolve(y_speech_cpu,y_rir_cpu,axes=2)[:,:,0:y_speech_.shape[2]],device='cuda')

            # metric loss
            gen_loss += self._metric_loss_speech(y_speech_, cs,mode=mode)
            gen_loss += self._metric_loss_reverb_speech(y_reverb_speech_, rs,mode=mode)
            gen_loss += self._metric_loss_rir(y_rir_, rir, self.filters,mode=mode)

            # adversarial loss
            if self.steps > self.discriminator_start:
                # p_ = self.model["discriminator"](y_)
                # if self.config["use_feat_match_loss"]:
                #     with torch.no_grad():
                #         p = self.model["discriminator"](x)
                # else:
                #     p = None
                # gen_loss += self._adv_loss(p_, p, mode=mode)

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
        if self.steps > self.discriminator_start:
            # re-compute y_ which leads better quality
            with torch.no_grad():
                # e = self.model["analyzer"].encoder(x)
                # z = self.model["analyzer"].projector(e)
                # zq, _, _ = self.model["analyzer"].quantizer(z)
                # y_ = self.model["generator"](zq)
                e_speech, e_rir = self.model["analyzer"].encoder(rs)
           

                z_speech = self.model["analyzer"].projector_speech(e_speech)
                zq_speech, _, _ = self.model["analyzer"].quantizer_speech(z_speech)

                z_rir = self.model["analyzer"].projector_rir(e_rir)
                zq_rir, _, _ = self.model["analyzer"].quantizer_rir(z_rir)

                y_speech_, y_rir_ = nn.parallel.data_parallel(self.model["generator"],(zq_speech,zq_rir),self.gpus)
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
            self._update_discriminator(dis_loss_speech,dis_loss_reverb)

        if(self.steps%5000==0 and self.steps>10):

            speech_path = os.path.join(self.save_path,"Speech_Vocoder")
            rir_path = os.path.join(self.save_path,"RIR_Vocoder")

            if(not os.path.exists(self.save_path)):
                os.mkdir(self.save_path)
    


            if(not os.path.exists(speech_path)):
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
        # e = self.model["analyzer"].encoder(x)
        # z = self.model["analyzer"].projector(e)
        # zq, _, _ = self.model["analyzer"].quantizer(z)
        # y_ = self.model["generator"](zq)

        e_speech, e_rir = self.model["analyzer"].encoder(rs)
           

        z_speech = self.model["analyzer"].projector_speech(e_speech)
        zq_speech, _, _ = self.model["analyzer"].quantizer_speech(z_speech)

        z_rir = self.model["analyzer"].projector_rir(e_rir)
        zq_rir, _, _ = self.model["analyzer"].quantizer_rir(z_rir)
            

        y_speech_, y_rir_ = nn.parallel.data_parallel(self.model["generator"],(zq_speech,zq_rir),self.gpus)

        y_speech_cpu = cp.asarray(y_speech_.to("cpu").detach())
        y_rir_cpu = cp.asarray(y_rir_.to("cpu").detach())
        y_reverb_speech_ = torch.tensor(fftconvolve(y_speech_cpu,y_rir_cpu,axes=2)[:,:,0:y_speech_.shape[2]],device='cuda')

        # metric loss
        gen_loss += self._metric_loss_speech(y_speech_, cs,mode=mode)
        gen_loss += self._metric_loss_reverb_speech(y_reverb_speech_, rs,mode=mode)
        gen_loss += self._metric_loss_rir(y_rir_, rir,self.filters,mode=mode)

        # adversarial loss & feature matching loss
        if self.steps > self.discriminator_start:

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

            # self._dis_loss_speech(p_speech_, p_speech, mode=mode)
            self._dis_loss_reverb(p_reverb_0, p_reverb0, mode=mode)
            self._dis_loss_reverb(p_reverb_1, p_reverb1, mode=mode)

        # generator loss
        self._record_loss('generator_loss', gen_loss, mode=mode)

        speech_path = os.path.join(self.save_path,"Speech_Eval_Vocoder")
        rir_path = os.path.join(self.save_path,"RIR_Eval_Vocoder")


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


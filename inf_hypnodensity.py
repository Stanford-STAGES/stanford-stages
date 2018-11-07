# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 23:58:10 2017

@author: jens
@modifer: hyatt
# from: inf_eval --> to: inf_generate_hypnodensity
"""
from decimal import *
import numpy as np
from inf_network import SCModel
from inf_config import ACConfig
import tensorflow as tf
import pyedflib
import time                     # for tracking time spent in encoding
import itertools                #  for extracting feature combinations
import pywt                     # wavelet entropy
import scipy.io as sio          # for noise level
import scipy.signal as signal   # for edf channel sampling and filtering
import os                       # for opening os files for pickle.

import pickle
import samplerate
from pathlib import Path

from inf_tools import myprint

import pdb

class Hypnodensity(object):

    def __init__(self,appConfig):
        self.config = appConfig
        self.hypnodensity = list()
        self.Features = HypnodensityFeatures(appConfig)
        self.CCsize = appConfig.CCsize

        self.channels = appConfig.channels
        self.channels_used =  appConfig.channels_used
        self.loaded_channels = appConfig.loaded_channels
        self.edf_pathname = appConfig.edf_path
        self.encodedD = []
        self.fs = appConfig.fs
        self.fsH = appConfig.fsH
        self.fsL = appConfig.fsL
        self.lightsOff = appConfig.lightsOff
        self.lightsOn = appConfig.lightsOn

        self.edf = [] #pyedflib.EdfFileReader

    def evaluate(self):
        p = Path(self.edf_pathname)
        p = Path(p.with_suffix('.pkl'))

        h = Path(self.edf_pathname)
        h = Path(h.with_suffix('.hypno_pkl'))

        if(p.exists()):

            myprint('Loading previously saved encoded data')
            with p.open('rb') as fp:
                self.encodedD = pickle.load(fp)
        else:
            myprint('Load EDF')
            self.loadEDF()
            myprint('Load noise level')
            self.psg_noise_level()
            print('Encode')
            self.encoding()

            # pickle our file
            with p.open('wb') as fp:
                pickle.dump(self.encodedD,fp)
                myprint("pickling done")

        if(h.exists()):
            myprint('Loading previously saved hynpodesnity')
            with h.open('rb') as fp:
                self.hypnodensity = pickle.load(fp)
        else:
            myprint('Score data')
            self.score_data()
            # pickle our file
            with h.open('wb') as fp:
                pickle.dump(self.hypnodensity,fp)
                myprint("Hypnodensity pickled")

    def get_hypnodensity(self):
        av = np.zeros(self.hypnodensity[0].shape)

        for i in range(len(self.hypnodensity)):
            av += self.hypnodensity[i]

        av = np.divide(av,len(self.hypnodensity))
        return av

    def get_features(self, modelName, idx):
        selected_features  = self.config.narco_prediction_selected_features
        X = self.Features.extract(self.hypnodensity[idx])
        X = self.Features.scale_features(X,modelName)
        return X[selected_features].T

    def extract_hjorth(self,x):

        #Segment
        N = np.arange(0,len(x),100*60*5)
        B = [np.expand_dims(x[N[i]:N[i+1]],1) for i in np.arange(len(N)-1)]
        B = np.concatenate(B,axis=1)
        #Activity
        act = np.var(B,axis=0)

        #Mobility

        mobil = self.mob(B)

        #Complexity
        comp = np.divide(self.mob(np.diff(B,axis=0)),mobil)

        #transform
        lAct = np.mean(np.log(act))
        lMobil = np.mean(np.log(mobil))
        lComp = np.mean(np.log(comp))


        return np.array([lAct,lMobil,lComp])

    def mob(self,B):
        diff = np.diff(B,axis=0)
        var = np.var(diff,axis=0)

        return np.sqrt(np.divide(var,np.var(B,axis=0)))

    def encoding(self):

        count = -1
        enc = []
        # Central, Occipital, EOG and chin

        numIterations = 4;  # there are actually 5 for CC, but this is just for displaying progress
        numConcatenates = 5;
        pdb.set_trace()
        for c in self.channels: # ['C3','C4','O1','O2','EOG-L','EOG-R','EMG','A1','A2']
            start_time = time.time()

            if isinstance(self.channels_used[c], int ):
                count += 1
                n  = int(self.fs*self.CCsize[c])
                p  = int(self.fs*(self.CCsize[c]-0.25))

                B1 = self.buffering(self.loaded_channels[c],n,p)
                print(B1.shape)
                zeroP = np.zeros([int(B1.shape[0]/2),B1.shape[1]])
                B1 = np.concatenate([zeroP,B1,zeroP],axis=0)

                n  = int(self.fs*self.CCsize[c]*2)
                p  = int(self.fs*(self.CCsize[c]*2-0.25))

                B2 = self.buffering(np.concatenate([np.zeros(int(n/4)),
                            self.loaded_channels[c],np.zeros(int(n/4))]),n,p)

                B2 = B2[:,:B1.shape[1]]

                start_time = time.time()

                F = np.fft.fft(B1,axis=0)
                C = np.conj(np.fft.fft(B2,axis=0))

                elapsed_time = time.time()-start_time
                myprint("Finished FFT  %d of %d\nTime elapsed = %0.2f"%(count+1,numIterations,elapsed_time));
                start_time = time.time()

                CC = np.real(np.fft.fftshift(np.fft.ifft(np.multiply(F,C),axis=0),axes=0))

                elapsed_time = time.time()-start_time
                myprint("Finished CC %d of %d\nTime elapsed = %0.2f"%(count+1,numIterations,elapsed_time));
                start_time = time.time()

                CC[np.isnan(CC)] = 0
                CC[np.isinf(CC)] = 0

                CC = CC[int(CC.shape[0]/4):int(CC.shape[0]*3/4),:]
                sc = np.max(CC,axis=0)
                sc = np.multiply(np.sign(sc),np.log((np.abs(sc)+1)/
                                 (self.CCsize[c]*self.fs)))/(sc+1e-10)

                CC = np.multiply(CC,sc)
                CC.astype(np.float32)

                if len(enc)>0:
                    enc = np.concatenate([enc,CC])
                else:
                    enc = CC

                if count==2:
                    eog1 = F
                if count==3:
                    PS = eog1*C
                    CC = np.real(np.fft.fftshift(np.fft.ifft(PS,axis=0),axes=0))
                    CC = CC[int(CC.shape[0]/4):int(CC.shape[0]*3/4),:]
                    sc = np.max(CC,axis=0)
                    sc = np.multiply(np.sign(sc),np.log((np.abs(sc)+1)/
                                     (self.CCsize[c]*self.fs)))/(sc+1e-10)
                    CC = np.multiply(CC,sc)
                    CC.astype(np.float32)


                    enc = np.concatenate([enc,CC])

                # pdb.set_trace()
                elapsed_time = time.time()-start_time
                myprint("Finished enc concatenate %d of %d\nTime elapsed = %0.2f"%(count+1,numConcatenates,elapsed_time))

        self.encodedD = enc

        if isinstance(self.lightsOff,int):
            self.encodedD = self.encodedD[:,4*30*self.lightsOff:4*30*self.lightsOn]  # This does not make sense ...


    def buffering(self,x, n, p=0):

        if p >= n:
            raise ValueError('p ({}) must be less than n ({}).'.format(p,n))

        # Calculate number of columns of buffer array
        cols = int(np.floor(len(x)/(n-p)))
        # Check for opt parameters

        # Create empty buffer array
        b = np.zeros((n, cols))

        # Fill buffer by column handling for initial condition and overlap
        j = 0
        slide = n - p
        start = 0
        for i in range(cols-int(np.ceil(n/(n-p)))):
            # Set first column to n values from x, move to next iteration
            b[:,i] = x[start:start+n]
            start += slide

        return b

    def loadEDF(self):
        if not self.edf:

            try:
                self.edf = pyedflib.EdfReader(self.edf_pathname)
            except OSError as osErr:
                print("OSError:","Loading",self.edf_pathname)
                raise(osErr)

        for c in self.channels: # ['C3','C4','O1','O2','EOG-L','EOG-R','EMG','A1','A2']
            myprint('Loading', c)
            if isinstance(self.channels_used[c], int ):

                self.loaded_channels[c] = self.edf.readSignal(self.channels_used[c])
                if self.edf.getPhysicalDimension(self.channels_used[c]).lower()=='mv':
                    myprint('mv')
                    self.loaded_channels[c] *= 1e3
                elif self.edf.getPhysicalDimension(self.channels_used[c]).lower()=='v':
                    myprint('v')
                    self.loaded_channels[c] *= 1e6

                #myprint('Resampling skipped ...')

                fs = self.edf.samplefrequency(self.channels_used[c])
                fs = Decimal(fs).quantize(Decimal('.0001'), rounding=ROUND_DOWN)
                print('fs',fs)

                self.resampling(c,fs)
                print('Resampling done')

                self.filtering(c,100)
                print('filtering done')

            else:
                print('channel[',c,'] was empty (skipped)',sep='');

    def loadHeader(self):
        if not self.edf:
            print(self.edf_pathname)
            self.edf = pyedflib.EdfReader(self.edf_pathname)

        signal_labels = self.edf.getSignalLabels()
        return signal_labels

    def filtering(self,c,fs):
        if fs>(2*self.fsL):
            Fl = signal.butter(5, self.fsL/(fs/2), btype='lowpass', output='sos')
            self.loaded_channels[c] = signal.sosfiltfilt(Fl, self.loaded_channels[c],
                                axis=-1, padtype='odd', padlen=None)

        Fh = signal.butter(5, self.fsH/(fs/2), btype='highpass', output='sos')
        self.loaded_channels[c] = signal.sosfiltfilt(Fh, self.loaded_channels[c],
                            axis=-1, padtype='odd', padlen=None)

    def resampling(self,c,fs):
        #ratio = np.float(self.fs)/np.round(np.float(fs));
        myprint("original samplerate = ",fs);
        myprint("resampling to ",self.fs)
        self.loaded_channels[c] = signal.resample_poly(self.loaded_channels[c],
                            self.fs, fs, axis=0, window=('kaiser', 5.0))
        #[N,D] = rat(desired_samplerate/src_samplerate);
        #if N!=D:
        #    if len(self.loaded_channels[c])>0:
        #        raw_data = resample(raw_data,N,D); #%resample to get the desired sample rate

        #print('ratio')
        #converter = 'sinc_best'
        #self.loaded_channels[c] = samplerate.resample(self.loaded_channels[c], ratio, converter)


        # self.loaded_channels[c] = samplerate.resample(self.loaded_channels[c], ratio, converter)
        #resampler = samplerate.Resampler(converter, channels=1)
        #print('resampler')

        #self.loaded_channels[c] = resampler.process(self.loaded_channels[c],
        #                                     ratio, end_of_input=True)
        #print('resampler.process')
        #self.loaded_channels[c] = signal.resample_poly(self.loaded_channels[c],
        #                    self.fs, fs, axis=0, window=('kaiser', 5.0))

    def psg_noise_level(self):
        noiseM = sio.loadmat(self.config.psg_noise_file_pathname, squeeze_me = True, struct_as_record = False)
        noiseM = noiseM['noiseM']

        noise = np.ones(4)*np.inf
        count = -1
        for c in self.channels[:4]:
            count += 1

            if self.channels_used[c]:
                hjorth = self.extract_hjorth(self.loaded_channels[c])

                cov = np.array(noiseM.covM[count])

                covI = np.linalg.inv(cov)
                meanV = np.array(noiseM.meanV[count])
                noise[count] = np.sqrt(np.matmul(np.matmul(np.transpose(hjorth-meanV),
                     covI),(hjorth-meanV)))

        notUsedC = np.argmax(noise[:2])
        notUsedO = np.argmax(noise[2:4])+2

        self.channels_used[self.channels[notUsedC]] = []
        self.channels_used[self.channels[notUsedO]] = []

    def softmax(self,x):

        #e_x = np.exp(x - np.max(x))
        #return e_x / e_x.sum()
        e_x = np.exp(x)
        div = np.repeat(np.expand_dims(np.sum(e_x,axis=1),1),5,axis=1)
        return np.divide(e_x,div)


    def score_data(self):
        self.hypnodensity = list()
        for l in self.config.models_used:
            hyp = Hypnodensity.run_data(self.encodedD,l,self.config.hypnodensity_model_root_path)
            pdb.set_trace()
            hyp = self.softmax(hyp)
            self.hypnodensity.append(hyp)

    def run_data(dat,model, root_model_path):

        ac_config = ACConfig(model_name=model, is_training=False, root_model_dir = root_model_path)
        #root_train_data_dir,
        #root_test_data_dir))
        hyp = Hypnodensity.run(dat,ac_config)
        return hyp

    def segment(dat,ac_config):
        # Get integer value for segment size using //
        n_seg = dat.shape[1]//ac_config.segsize

        #For debugging
        pdb.set_trace()

        # Incorrect I think ... commented out on 10/30/2018  @hyatt
        # dat = np.expand_dims(dat[:n_seg*ac_config.segsize,:],0)
        dat = np.expand_dims(dat[:,:n_seg*ac_config.segsize],0)

        num_batches = np.int(np.ceil(np.divide(dat.shape[2],(ac_config.eval_nseg_atonce*ac_config.segsize),dtype='float')))

        Nextra = np.int(np.ceil(num_batches * ac_config.eval_nseg_atonce * ac_config.segsize)%dat.shape[2])
              # why not:    Nextra = num_batches * ac_config.eval_nseg_atonce * ac_config.segsize - dat.shape[2]

        # fill remaining (nExtra) values with the mean value of each column
        meanF = np.mean(np.mean(dat,2),0) * np.ones([1,Nextra,dat.shape[1]])


        dat = np.transpose(dat, [0, 2, 1])
        dat = np.concatenate([dat,meanF],1)

        prediction = np.zeros([num_batches*ac_config.eval_nseg_atonce,5])

        return dat, Nextra, prediction, num_batches

    def run(dat,ac_config):

         with tf.Graph().as_default() as g:

            m = SCModel(ac_config)
            s = tf.train.Saver(tf.global_variables())
            # print("AC config hypnodensity path",ac_config.hypnodensity_model_dir)
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as session:

                ckpt = tf.train.get_checkpoint_state(ac_config.hypnodensity_model_dir)
                #For debugging
                # pdb.set_trace()

                s.restore(session, ckpt.model_checkpoint_path)

                state = np.zeros([1,ac_config.num_hidden*2])

                dat, Nextra, prediction, num_batches = Hypnodensity.segment(dat, ac_config)
                for i in range(num_batches):

                    x = dat[:,i*ac_config.eval_nseg_atonce*ac_config.segsize:(i+1)*ac_config.eval_nseg_atonce*ac_config.segsize,:]

                    est, _ = session.run([m.logits, m.final_state], feed_dict={
                            m.features: x,
                            m.targets: np.ones([ac_config.eval_nseg_atonce*ac_config.segsize,5]),
                            m.mask: np.ones(ac_config.eval_nseg_atonce*ac_config.segsize),
                            m.batch_size: np.ones([1]),
                            m.initial_state: state
                            })

                    prediction[i*ac_config.eval_nseg_atonce:(i+1)*ac_config.eval_nseg_atonce,:] = est

                prediction = prediction[:-int(Nextra/ac_config.segsize),:]

                return prediction

class HypnodensityFeatures(object):  # <-- extract_features

    def __init__(self,appConfig):
        self.config = appConfig
        self.meanV = []
        self.scaleV = []
        self.selected = [] #[1, 11, 16, 22, 25, 41, 43, 49, 64, 65, 86, 87, 103, 119, 140, 147, 149, 166, 196, 201, 202, 220, 244, 245, 261, 276, 289, 296, 299, 390, 405, 450, 467, 468, 470, 474, 476, 477]
        self.scale_path = appConfig.hypnodensity_scale_path # 'scaling'
        self.select_features_path = appConfig.hypnodensity_select_features_path
        self.select_features_pickle_name = appConfig.hypnodensity_select_features_pickle_name # 'narcoFeatureSelect.p'



    def extract(self,hyp):
        eps = 1e-10
        features = np.zeros([24+31*15])
        j = -1
        f = 10

        for i in range(5):
            for comb in itertools.combinations([0,1,2,3,4],i+1):
                j +=1
                dat = np.prod(hyp[:,comb],axis=1)**(1/float(len(comb)))

                features[j*15] = np.log(np.mean(dat)+eps)
                features[j*15+1] = -np.log(1-np.max(dat))

                moving_av = np.convolve(dat,np.ones(10),mode='valid')
                features[j*15+2] = np.mean(np.abs(np.diff(moving_av)))

                features[j*15+3] = self.wavelet_entropy(dat) #Shannon entropy - check if it is used as a feature

                rate = np.cumsum(dat)/np.sum(dat)
                I1 = (i for i,v in enumerate(rate) if v>0.05).__next__()
                features[j*15+4] = np.log(I1*2+eps)
                I2 = (i for i,v in enumerate(rate) if v>0.1).__next__()
                features[j*15+5] = np.log(I2*2+eps)
                I3 = (i for i,v in enumerate(rate) if v>0.3).__next__()
                features[j*15+6] = np.log(I3*2+eps)
                I4 = (i for i,v in enumerate(rate) if v>0.5).__next__()
                features[j*15+7] = np.log(I4*2+eps)

                features[j*15+8] = np.sqrt(np.max(dat)*np.mean(dat)+eps)
                features[j*15+9] = np.mean(np.abs(np.diff(dat))*np.mean(dat)+eps)
                features[j*15+10] = np.log(self.wavelet_entropy(dat)*np.mean(dat)+eps)
                features[j*15+11] = np.sqrt(I1*2*np.mean(dat))
                features[j*15+12] = np.sqrt(I2*2*np.mean(dat))
                features[j*15+13] = np.sqrt(I3*2*np.mean(dat))
                features[j*15+14] = np.sqrt(I4*2*np.mean(dat))

        rem = (hyp.shape[0]%2)
        if rem==1:
            data =  hyp[:-rem,:]
        else:
            data = hyp

        data = data.reshape([-1,2,5])
        data = np.squeeze(np.mean(data,axis=1))

        S = np.argmax(data,axis=1)

        SL = [i for i,v in enumerate(S) if v!=0]
        if len(SL)==0:
            SL = len(data)
        else:
            SL = SL[0]

        RL = [i for i,v in enumerate(S) if v==4]
        if len(RL)==0:
            RL = len(data)
        else:
            RL = RL[0]


        # Nightly SOREMP

        wCount = 0;
        rCount = 0;
        rCountR = 0;
        soremC = 0;
        for i in range(SL,len(S)):
            if (S[i]==0) | (S[i]==1):
                wCount += 1
            elif (S[i]==4) & (wCount>4):
                rCount += 1
                rCountR += 1
            elif rCount>1:
                soremC += 1
            else:
                wCount = 0
                rCount = 0

        # NREM Fragmentation
        nCount = 0
        nFrag = 0
        for i in range(SL,len(S)):
            if (S[i]==2) | (S[i]==3):
                nCount += 1
            elif ((S[i]==0) | (S[i]==1)) & (nCount>3):
                nFrag += 1
                nCount = 0

        # W/N1 Bouts
        wCount = 0
        wBout = 0
        wCum = 0
        sCount = 0
        for i in range(SL,len(S)):
            if S[i]!=1:
                sCount += 1

            if (sCount>5) & ((S[i]==0) | (S[i]==1)):
                wCount = wCount+1
                if wCount<30:
                    wCum = wCum+1

            elif wCount>4:
                wCount = 0
                wBout = wBout+1

        #

        features[-24] = self.logmodulus(SL*f)
        features[-23] = self.logmodulus(RL-SL*f)

        features[-22] = np.sqrt(rCountR)
        features[-21] = np.sqrt(soremC)
        features[-20] = np.sqrt(nFrag)
        features[-19] = np.sqrt(wCum)
        features[-18] = np.sqrt(wBout)

        ## Find out what features are used:...!
        features[-17:] = self.logmodulus(self.transitionFeatures(data))

        return features

    def select_features(self,threshold = 1):

        if len(self.selected)==0:
            try:
                with open(os.path.join(self.select_features_path,self.select_features_pickle_name), 'rb') as sel:
                    S = pickle.load(sel)
                    self.selected = S>threshold
            except FileNotFoundError as e:
                print("File not found ",e)

        return self.selected

    def logmodulus(self,x):
        return np.sign(x)*np.log(abs(x)+1)

    def scale_features(self,features,sc_mod = 'unknown'):
        scaled_features = features
        if len(scaled_features.shape)==1:
            scaled_features = np.expand_dims(scaled_features,axis=1)

        if len(self.meanV)==0:
            try:
                with open(os.path.join(self.scale_path, sc_mod + '_scale.p'), 'rb') as sca:
                    scaled = pickle.load(sca)
                self.meanV = np.expand_dims(scaled['meanV'],axis=1)[:,:,0]
                self.scaleV = np.expand_dims(scaled['scaleV'],axis=1)[:,:,0]
            except FileNotFoundError as e:
                print("File not found ",e)
                print("meanV set to 0 and scaleV set to 1")
                self.meanV = 0;
                self.scaleV = 1;



        scaled_features -= self.meanV
        scaled_features = np.divide(scaled_features,self.scaleV)

        scaled_features[scaled_features>10] = 10
        scaled_features[scaled_features<-10] = -10

        return scaled_features

    def transitionFeatures(self,data):
        S = np.zeros(data.shape)
        for i in range(5):
            S[:,i] = np.convolve(data[:,i],np.ones(9),mode='same')

        S = self.softmax(S)

        cumR = np.zeros(S.shape)
        Th = 0.2;
        peakTh = 10;
        for j in range(5):
            for i in range(len(S)):
                if S[i-1,j]>Th:
                    cumR[i,j] = cumR[i-1,j] + S[i-1,j]

            cumR[cumR[:,j]<peakTh,j] = 0

        for i in range(5):
            d = cumR[:,i]
            indP = self.find_peaks(cumR[:,i])
            typeP = np.ones(len(indP))*i
            if i==0:
                peaks = np.concatenate([np.expand_dims(indP,axis=1),np.expand_dims(typeP,axis=1)],axis=1)
            else:
                peaks = np.concatenate([peaks,np.concatenate([np.expand_dims(indP,axis=1),
                                                              np.expand_dims(typeP,axis=1)],axis=1)],axis=0)

        I = [i[0] for i in sorted(enumerate(peaks[:,0]), key=lambda x:x[1])]
        peaks = peaks[I,:]

        remList = np.zeros(len(peaks))

        peaks[peaks[:,1]==0,1] = 1
        peaks[:,1] = peaks[:,1]-1

        if peaks.shape[0]<2:
            features = np.zeros(17)
            return features

        for i in range(peaks.shape[0]-1):
            if peaks[i,1]==peaks[i+1,1]:
                peaks[i+1,0] += peaks[i,0]
                remList[i] = 1
        remList = remList==0
        peaks = peaks[remList,:]
        transitions = np.zeros([4,4])

        for i in range(peaks.shape[0]-1):
            transitions[int(peaks[i,1]),int(peaks[i+1,1])] = np.sqrt(peaks[i,0]*peaks[i+1,0])
        di = np.diag_indices(4)
        transitions[di] = None

        transitions = transitions.reshape(-1)
        transitions = transitions[np.invert(np.isnan(transitions))]
        nPeaks = np.zeros(5)
        for i in range(4):
            nPeaks[i] = np.sum(peaks[:,1]==i)

        nPeaks[-1] = peaks.shape[0]

        features = np.concatenate([transitions,nPeaks],axis=0)
        return features

    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x)
        div = np.repeat(np.expand_dims(np.sum(e_x,axis=1),1),5,axis=1)
        return np.divide(e_x,div)

    def find_peaks(self,x):
        peaks = []
        for i in range(1,len(x)):
            if x[i-1]>x[i]:
                peaks.append(i-1)

        return np.asarray(peaks)

    def wavelet_entropy(self,dat):
        coef, freqs = pywt.cwt(dat,np.arange(1,60), 'gaus1')
        Eai = np.sum(np.square(np.abs(coef)),axis=1)
        pai = Eai/np.sum(Eai)

        WE = -np.sum(np.log(pai)*pai)

        return WE

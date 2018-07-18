# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 00:26:55 2017

@author: jens
"""
import pdb
import numpy as np
import scipy.signal as signal
import pywt
import itertools
import scipy.io as sio
import os
import pickle

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

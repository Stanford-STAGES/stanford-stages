"""
@author: jens
@modifiers: hyatt, neergaard

Migrated from inf_hypnodensity on 12/6/2019
"""
import pickle
import numpy as np
import pywt  # wavelet entropy
import itertools  # for extracting feature combinations
import os  # for opening os files for pickle.
from inf_tools import softmax


class HypnodensityFeatures(object):  # <-- extract_features

    num_features = 489
    def __init__(self, app_config):
        self.config = app_config
        # Dictionaries, keyed by model names

        self.meanV = {}
        # Standard deviation of features.
        self.stdV = {}

        # range is calculated as difference between 15th and 85th percentile - this was previously the "scaleV".
        self.rangeV = {}
        self.medianV = {}

        try:
            self.selected = app_config.narco_prediction_selected_features
        except:
            self.selected = []  # [1, 11, 16, 22, 25, 41, 43, 49, 64, 65, 86, 87, 103, 119, 140, 147, 149, 166, 196, 201, 202, 220, 244, 245, 261, 276, 289, 296, 299, 390, 405, 450, 467, 468, 470, 474, 476, 477]

        self.scale_path = app_config.hypnodensity_scale_path  # 'scaling'

        # self.select_features_path = appConfig.hypnodensity_select_features_path
        # self.select_features_pickle_name = appConfig.hypnodensity_select_features_pickle_name  # 'narcoFeatureSelect.p'

    def extract(self, hyp):
        eps = 1e-10
        features = np.zeros([24 + 31 * 15])
        hyp = hyp[~np.isnan(hyp[:, 0]), :]  # or np.invert(np.isnan(hyp[:, 0])
        # k = [i for i, v in enumerate(hyp[:, 0]) if np.isnan(v)]
        # hyp[k[0] - 2:k[-1] + 2, :]
        j = -1

        for i in range(5):
            for comb in itertools.combinations([0, 1, 2, 3, 4], i + 1):
                j += 1
                dat = np.prod(hyp[:, comb], axis=1) ** (1 / float(len(comb)))

                features[j * 15] = np.log(np.mean(dat) + eps)
                features[j * 15 + 1] = -np.log(1 - np.max(dat))

                moving_av = np.convolve(dat, np.ones(10), mode='valid')
                features[j * 15 + 2] = np.mean(np.abs(np.diff(moving_av)))

                features[j * 15 + 3] = self.wavelet_entropy(dat)  # Shannon entropy - check if it is used as a feature

                rate = np.cumsum(dat) / np.sum(dat)
                try:
                    I1 = (i for i, v in enumerate(rate) if v > 0.05).__next__()
                except StopIteration:
                    I1 = len(hyp)
                features[j * 15 + 4] = np.log(I1 * 2 + eps)

                try:
                    I2 = (i for i, v in enumerate(rate) if v > 0.1).__next__()
                except StopIteration:
                    I2 = len(hyp)
                features[j * 15 + 5] = np.log(I2 * 2 + eps)

                try:
                    I3 = (i for i, v in enumerate(rate) if v > 0.3).__next__()
                except StopIteration:
                    I3 = len(hyp)
                features[j * 15 + 6] = np.log(I3 * 2 + eps)

                try:
                    I4 = (i for i, v in enumerate(rate) if v > 0.5).__next__()
                except StopIteration:
                    I4 = len(hyp)
                features[j * 15 + 7] = np.log(I4 * 2 + eps)

                features[j * 15 + 8] = np.sqrt(np.max(dat) * np.mean(dat) + eps)
                features[j * 15 + 9] = np.mean(np.abs(np.diff(dat)) * np.mean(dat) + eps)
                features[j * 15 + 10] = np.log(self.wavelet_entropy(dat) * np.mean(dat) + eps)
                features[j * 15 + 11] = np.sqrt(I1 * 2 * np.mean(dat))
                features[j * 15 + 12] = np.sqrt(I2 * 2 * np.mean(dat))
                features[j * 15 + 13] = np.sqrt(I3 * 2 * np.mean(dat))
                features[j * 15 + 14] = np.sqrt(I4 * 2 * np.mean(dat))

        rem = (hyp.shape[0] % 2)
        if rem == 1:
            data = hyp[:-rem, :]
        else:
            data = hyp

        data = data.reshape([-1, 2, 5])
        data = np.squeeze(np.mean(data, axis=1))

        S = np.argmax(data, axis=1)

        SL = [i for i, v in enumerate(S) if v != 0]
        if len(SL) == 0:
            SL = len(data)
        else:
            SL = SL[0]

        RL = [i for i, v in enumerate(S) if v == 4]
        if len(RL) == 0:
            RL = len(data)
        else:
            RL = RL[0]

        # Nightly SOREMP
        # rem within 30 minutes of sleep onset.  REM latency - Sleep latency = num 30 second epochs elapsed from sleep onsent to rem sleep onset; <= thirty 30s epochs is same as <= 15 minutes
        # has_sorem = (RL - SL) <= 30
        # features[-27] = RL

        wCount = 0
        rCount = 0
        rCountR = 0
        soremC = 0
        '''
        # The following was originally used, but found to be inconsistent with the described
         feature it implements.
        '''
        # for i in range(SL, len(S)):
        #     if (S[i] == 0) | (S[i] == 1):
        #         wCount += 1
        #     elif (S[i] == 4) & (wCount > 4):
        #         rCount += 1
        #         rCountR += 1
        #     elif rCount > 1:
        #         soremC += 1
        #     else:
        #         wCount = 0
        #         rCount = 0
        # features[-26] = np.sqrt(rCountR)
        # features[-25] = np.sqrt(soremC)

        '''
        Updated
        This ensures we meet the criteria for a SOREMP and also takes care of counting the first epoch of REM of 
        that SOREMP.  The manuscript code took care of the first epoch of REM but used too general of a description 
        for a SOREMP (i.e. missed the minimum requirement of one minute of REM). 
        '''
        wCount = 0
        rCount = 0
        rCountR = 0
        soremC = 0
        for i in range(SL, len(S)):
            if (S[i] == 0) | (S[i] == 1):
                wCount += 1
            elif (S[i] == 4) & (wCount > 4):
                rCount += 1
                if rCount == 2:
                    soremC += 1
                    rCountR += 2
                elif rCount > 2:
                    rCountR += 1
            else:
                wCount = 0
                rCount = 0

        # NREM Fragmentation
        nCount = 0
        nFrag = 0
        for i in range(SL, len(S)):
            if (S[i] == 2) | (S[i] == 3):
                nCount += 1
            elif ((S[i] == 0) | (S[i] == 1)) & (nCount > 3):
                nFrag += 1
                nCount = 0

        # W/N1 Bouts
        wCount = 0
        wBout = 0
        wCum = 0
        sCount = 0
        for i in range(SL, len(S)):
            # Used to just be S[i] != 1
            if S[i] != 1 and S[i] != 0:
                sCount += 1

            if (sCount > 5) and ((S[i] == 0) or (S[i] == 1)):
                wCount = wCount + 1
                if wCount < 30:
                    wCum = wCum + 1

            elif wCount > 4:
                wCount = 0
                wBout = wBout + 1

        features[-24] = self.logmodulus(SL)
        features[-23] = self.logmodulus(RL - SL)

        features[-22] = np.sqrt(rCountR)
        features[-21] = np.sqrt(soremC)
        features[-20] = np.sqrt(nFrag)
        features[-19] = np.sqrt(wCum)
        features[-18] = np.sqrt(wBout)

        ## Find out what features are used:...!
        features[-17:] = self.logmodulus(self.transition_features(data))
        return features

    def select_features(self, threshold=1):

        if len(self.selected) == 0:
            try:
                with open(os.path.join(self.select_features_path, self.select_features_pickle_name), 'rb') as sel:
                    S = pickle.load(sel)
                    self.selected = S > threshold
            except FileNotFoundError as e:
                print("File not found ", e)

        return self.selected

    # features is an Nx F array representing F features for N studies.  F should be self.num_features.
    # scale_method can be 'range', 'z', 'unscaled', or None.
    # if scale_method is None, then the configuration.narco_feature_scaling_method is used.
    def scale_features(self, features, sc_mod='unknown', scale_method=None):

        if scale_method is None:
            scale_method = self.config.narco_feature_scaling_method

        scaled_features = features
        if len(scaled_features.shape) == 1:
            scaled_features = np.expand_dims(scaled_features, axis=1)

        if sc_mod not in self.meanV:
            try:
                with open(os.path.join(self.scale_path, sc_mod + '_scale.p'), 'rb') as sca:
                    scaled = pickle.load(sca)

                self.meanV[sc_mod] = scaled['meanV'].reshape((1, self.num_features))
                self.stdV[sc_mod] = scaled['stdV'].reshape((1, self.num_features))
                self.medianV[sc_mod] = scaled['medianV'].reshape((1, self.num_features))
                self.rangeV[sc_mod] = scaled['rangeV'].reshape((1, self.num_features))
                # scale_v = scaled['scaleV'].reshape((1, -1))  # same thing provided there are self.num_features values for scaleV as well.

            except FileNotFoundError as e:
                print("File not found ", e)
                print("offsetV set to 0 and scaleV set to 1")
                self.meanV[sc_mod] = self.medianV[sc_mod]  = 0
                self.stdV[sc_mod] = self.rangeV[sc_mod] = 1

        if scale_method == 'range':
            offset_v = self.medianV[sc_mod]
            scale_v = self.rangeV[sc_mod]
        elif scale_method == 'z':
            offset_v = self.meanV[sc_mod]
            scale_v = self.stdV[sc_mod]
        else:
            offset_v = 0
            scale_v = 1

        if np.any(scale_v == 0):
            print(
                f'Warning:  Found a 0 scale value for {sc_mod}.  Divide by 0 to follow.')  # Setting to 1 to avoid divide by 0.')
            # self.scaleV[sc_mod] = 1

        scaled_features -= offset_v
        scaled_features = np.divide(scaled_features, scale_v)

        if scale_method != 'unscaled':
            scaled_features[scaled_features > 10] = 10
            scaled_features[scaled_features < -10] = -10

        print(scale_method, 'method applied for scaling')

        # For debugging:  How many are less than 10 -->  (scaled_features < -10).sum()
        return scaled_features

    @staticmethod
    def transition_features(data):
        S = np.zeros(data.shape)
        for i in range(5):
            S[:, i] = np.convolve(data[:, i], np.ones(9), mode='same')

        S = softmax(S)

        cumR = np.zeros(S.shape)
        Th = 0.2
        peakTh = 10
        for j in range(5):
            for i in range(len(S)):
                if S[i - 1, j] > Th:
                    cumR[i, j] = cumR[i - 1, j] + S[i - 1, j]

            cumR[cumR[:, j] < peakTh, j] = 0

        for i in range(5):
            d = cumR[:, i]
            indP = HypnodensityFeatures.find_peaks(cumR[:, i])
            typeP = np.ones(len(indP)) * i
            if i == 0:
                peaks = np.concatenate([np.expand_dims(indP, axis=1), np.expand_dims(typeP, axis=1)], axis=1)
            else:
                peaks = np.concatenate([peaks, np.concatenate([np.expand_dims(indP, axis=1),
                                                               np.expand_dims(typeP, axis=1)], axis=1)], axis=0)

        I = [i[0] for i in sorted(enumerate(peaks[:, 0]), key=lambda x: x[1])]
        peaks = peaks[I, :]

        remList = np.zeros(len(peaks))

        peaks[peaks[:, 1] == 0, 1] = 1
        peaks[:, 1] = peaks[:, 1] - 1

        if peaks.shape[0] < 2:
            features = np.zeros(17)
            return features

        for i in range(peaks.shape[0] - 1):
            if peaks[i, 1] == peaks[i + 1, 1]:
                peaks[i + 1, 0] += peaks[i, 0]
                remList[i] = 1
        remList = remList == 0
        peaks = peaks[remList, :]
        transitions = np.zeros([4, 4])

        for i in range(peaks.shape[0] - 1):
            transitions[int(peaks[i, 1]), int(peaks[i + 1, 1])] = np.sqrt(peaks[i, 0] * peaks[i + 1, 0])
        di = np.diag_indices(4)
        transitions[di] = None

        transitions = transitions.reshape(-1)
        transitions = transitions[np.invert(np.isnan(transitions))]
        nPeaks = np.zeros(5)
        for i in range(4):
            nPeaks[i] = np.sum(peaks[:, 1] == i)

        nPeaks[-1] = peaks.shape[0]

        features = np.concatenate([transitions, nPeaks], axis=0)
        return features

    @staticmethod
    def logmodulus(x):
        return np.sign(x) * np.log(abs(x) + 1)

    @staticmethod
    def find_peaks(x):
        peaks = []
        for i in range(1, len(x)):
            if x[i - 1] > x[i]:
                peaks.append(i - 1)

        return np.asarray(peaks)

    @staticmethod
    def wavelet_entropy(dat):
        coef, freqs = pywt.cwt(dat, np.arange(1, 60), 'gaus1')
        Eai = np.sum(np.square(np.abs(coef)), axis=1)
        pai = Eai / np.sum(Eai)

        WE = -np.sum(np.log(pai) * pai)
        return WE

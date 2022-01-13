# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 23:58:10 2017

@author: jens
@modifier: hyatt
@modifier: neergaard
# from: inf_eval --> to: inf_generate_hypnodensity
"""
import json
import traceback
import pickle
import time  # for auditing code speed.
from pathlib import Path

import h5py  # Adding h5py support :) - requiring h5py support :(
import numpy as np
import pyedflib
import scipy.io as sio  # for noise level
import scipy.signal as signal  # for edf channel sampling and filtering

import tensorflow as tf
from pandas import read_csv
from scipy.fftpack import fft, ifft, fftshift
from inf_config import ACConfig
from inf_narco_features import HypnodensityFeatures
from inf_network import SCModel
from inf_tools import myprint, softmax, rolling_window_nodelay, StanfordStagesError
from pkg_resources import resource_filename


def config_property(name):
    @property
    def prop(self):
        return getattr(self.config, name, None)

    return prop


class Hypnodensity(object):
    edf_filename = config_property('edf_filename')
    channels = config_property('channels')
    channels_used = config_property('channels_used')
    loaded_channels = config_property('loaded_channels')
    lights_on = config_property('lights_on')
    lights_off = config_property('lights_off')
    fs_high = config_property('fs_high')
    fs_low = config_property('fs_low')
    cc_size = config_property('cc_size')

    def __init__(self, app_config):
        self.config = app_config
        self.hypnodensity = list()

        self._hypnodensity_features = {}
        self.flatline = []
        self.features = HypnodensityFeatures(app_config)
        self.edf: pyedflib.EdfFileReader = []

        '''
        encoded_data is a 1640 x N array (np.array)
        Rows represent stacked cross correlation values.
        The order is: Central, Occipital, EOG-L, EOG-R, EOG-L/R, and then chin
        The size of the cross correlation value is specied by app_config.cc_ssize, with a default of
        2s (central), 2s (occipital), 4s, 4s, 4s (eog's), and 0.4s (chin).
        Each column, represents a 0.250 s lapse or shift in time from the start of the psg (i.e. t0 = 0.0s)
        Because the psg channels are resampled to 100 Hz, the PSG channels can be sliced as follows:
         Central - 0:199
         Occipital - 200:399
         EOG-L - 400:799
         EOG-R - 800:1199
         EOG-L/R - 1200:1599
         Chin - 1600:1639
        N can be determined as follows:
        N = (D-(max(cc_size)-delta_lapse))/delta_lapse
            where D is the maximum duration of the PSG in seconds which can be divided by 30 without giving a remainder,
            max(cc_size) is 4 s, and delta_lapse is 0.25s.
        N = (D-3.75)*4
        Thus if a PSG is 30980s, then D = 30960 and N = 123825
        '''
        self.encoded_data = []
        self._encoded_data_channel_slices = {'central': range(0, 200),
                                             'occipital': range(200, 400),
                                             'eog-l': range(400, 800),
                                             'eog-r': range(800, 1200),
                                             'eog-lr': range(1200, 1600),
                                             'chin': range(1600, 1640)}

        self._encoded_data_channel_offsets = {'central': 0,
                                              'occipital': 200,
                                              'eog-l': 400,
                                              'eog-r': 800,
                                              'eog-lr': 1200,
                                              'chin': 1600}

        self.fs = int(app_config.fs)

        # Filter specifications for resampling from MATLAB
        try:
            filter_specs_path = resource_filename('_resources', 'filter_specs.json')
            with open(filter_specs_path, "r") as json_file:
                self.filter_specs = json.load(json_file)
        except FileNotFoundError:
            self.filter_specs = {}
            myprint('Unable to load filter specifications file.  Default resampling filter coefficients will be used'
                    ' instead.')

    def audit(self, method_to_audit, audit_label, *args):
        start_time = time.time()
        method_to_audit(*args)
        elapsed_time = time.time() - start_time
        with Path(self.config.filename['audit']).open('a') as fp:
            audit_str = f', {audit_label}: {elapsed_time:0.3f} s'
            fp.write(audit_str)

    def export_features(self, p=None):
        _features = self._hypnodensity_features
        if not isinstance(p, Path):
            p = Path(p)
        if isinstance(p, Path):
            try:
                in_a_pickle = p.suffix != '.h5'  # p.suffix == '.pkl'
                if in_a_pickle:
                    with p.open('wb') as fp:
                        pickle.dump(_features, fp)
                else:
                    with h5py.File(p, 'w') as fp:
                        for model, values in _features.items():
                            fp[model] = values
                print(f'Hypnodensity features saved to "{str(p)}"')
                return True
            except:
                print(f'EXCEPTION caught while saving hypnodensity features to {str(p)}. FAIL.\n')
                traceback.print_exc()
                return False
        else:
            print('Not an instance of Path')
            return False

    # Returns the imported hypnodensity model features on success.  Otherwise returns None
    def import_model_features(self, p=None, model=None):
        _features = None
        if isinstance(p, Path) and p.exists():
            self.myprint('Loading previously saved hypnodensity features')
            in_a_pickle = p.suffix != '.h5'  # p.suffix == '.pkl'
            if in_a_pickle:
                with p.open('rb') as fp:
                    _features = pickle.load(fp)
            else:
                _features = {}
                with h5py.File(p, 'r') as fp:
                    for key, value in fp.items():
                        _features[key] = value[()]
                    #_features = {k:fp[k][()] for k in fp.keys()}

            if model is None:
                self._hypnodensity_features = _features
            elif model in _features:
                self._hypnodensity_features[model] = _features[model]
                _features = _features[model]
            else:
                print('import_features requires model name, when provided as an argument, to exist as a key in the imported file.  A model name was provided but not found in the import file; so nothing imported :(')
        return _features

    def export_hypnodensity(self, p=None):
        if isinstance(p, Path):
            in_a_pickle = p.suffix != '.h5'  # p.suffix == '.pkl'
            if in_a_pickle:
                with p.open('wb') as fp:
                    pickle.dump(self.hypnodensity, fp)
                    self.myprint("Hypnodensity pickled")
            else:
                with h5py.File(p, 'w') as fp:
                    fp['hypnodensity'] = self.hypnodensity
            return True
        else:
            print('Not an instance of Path')
            return False

    def import_hypnodensity(self, p=None):
        if isinstance(p, Path) and p.exists():
            self.myprint(f'Loading previously saved hypnodensity ({p.suffix})')
            in_a_pickle = p.suffix != '.h5'  # p.suffix == '.pkl'
            _hypno = []
            if in_a_pickle:
                with p.open('rb') as fp:
                    _hypno = pickle.load(fp)
            else:
                with h5py.File(p, 'r') as fp:
                    _hypno = fp['hypnodensity'][()]

            if not isinstance(_hypno, list):
                if _hypno.ndim == 3:
                    pass
                    # _hypno = _hypno.tolist()
                elif _hypno.ndim == 2:
                    _hypno = [_hypno]
                else:
                    print('Warning: Unsupported number of dimensions found for hypnodensity:', _hypno.ndim)

            self.hypnodensity = _hypno
            return True
        else:
            return False

    def export_encoded_data(self, p=None):
        if isinstance(p, Path):
            in_a_pickle = p.suffix != '.h5'  # p.suffix == '.pkl'
            if in_a_pickle:
                print(f"Pickle encode data to: {p}")
                with p.open('wb') as fp:
                    pickle.dump(self.encoded_data, fp)
                    pickle.dump(self.channels_used, fp)
                print(f"Encode data pickled to: {p}\n")
            else:
                # print("Not in a pickle!")
                with h5py.File(p, 'w') as fp:
                    fp['encodedD'] = self.encoded_data
                    fp['channels_used'] = np.array(list(self.channels_used.keys()), dtype='S')
                self.myprint(".h5 exporting done")
            return True
        else:
            print('Not an instance of Path')
            return False

    def import_encoded_data(self, p=None):
        if isinstance(p, Path) and p.exists():
            in_a_pickle = p.suffix != '.h5'  # p.suffix == '.pkl'
            self.myprint('Loading previously saved encoded data')
            if in_a_pickle:
                with p.open('rb') as fp:
                    self.encoded_data = pickle.load(fp)
            else:
                with h5py.File(p, 'r') as fp:
                    self.encoded_data = fp['encodedD'][()]
            return True
        else:
            return False

    def evaluate(self):
        # Determine if we are caching and/or have cached results
        p = None
        if self.config.saveEncoding:
            p = Path(self.config.encodeFilename)

        h = Path(self.config.filename["hypnodensity_h5"])
        # h = Path(self.config.filename["hypnodensity_pkl"])

        is_auditing = self.config.filename['audit'] is not None
        audit_hypnodensity = is_auditing and self.config.audit['hypnodensity']
        audit_encoding = is_auditing and (self.config.filename["h5_encoding"] is not None
                                          or self.config.filename["pkl_encoding"] is not None)

        # The goal is to generate the hypnodensity or audit the steps along the way
        # if audit_hypnodensity -> then don't import_hypnodensity
        # or if audit_encoding -> then don't bypass the encoding step
        # or if not self.import_hypnodensity -> then go through the steps to create it as follows
        if audit_hypnodensity or audit_encoding or not self.import_hypnodensity(h):

            # Go through all the steps if we are doing an audit
            if audit_encoding:
                self.audit(self.loadEDF, 'Load EDF')
                self.audit(self.psg_noise_level, 'Calculating noise levels')
                self.audit(self.filtering, 'Channel filter')
                self.audit(self.encoding, 'Encoding')
                if self.config.saveEncoding:
                    if self.config.filename["h5_encoding"] is not None:
                        h5_path = Path(self.config.filename['h5_encoding'])
                        self.audit(self.export_encoded_data, 'export encoding as .h5', h5_path)
                        self.audit(self.import_encoded_data, '.h5 import encoding', h5_path)
                    if self.config.filename["pkl_encoding"] is not None:
                        pkl_path = Path(self.config.filename['pkl_encoding'])
                        self.audit(self.export_encoded_data, 'export encoding as .pkl', pkl_path)
                        self.audit(self.import_encoded_data, '.pkl import encoding', pkl_path)

            # Otherwise go ahead and try to import the previously encoded data ...
            elif not self.import_encoded_data(p):
                self.encode_edf(p)

            # If we are just encoding the file for future use, then we don't want to spend time running the models right
            # now and can skip this part.  Otherwise if we are auditing or not able to import a cached hypnodensity,
            # then we want to generate the hypnodensity
            if audit_hypnodensity or not self.config.encodeOnly:
                if not self.import_hypnodensity(h):
                    print('Calculating hypnodensity')
                    if audit_hypnodensity:
                        self.audit(self.score_data, 'Generate hypnodensity')
                    else:
                        self.score_data()
                    # cache our file
                    if self.config.save_hypnodensity_h5:
                        self.export_hypnodensity(h)

    def encode_edf(self, export_path=None):
        # and if you can't then go through all the steps to encode it
        self.myprint('Load EDF')
        self.loadEDF()
        self.myprint('Load noise level')
        self.psg_noise_level()
        print('Filtering channels')
        self.filtering()
        print('Encoding')
        self.encoding()
        print('Encoding done')
        if self.config.saveEncoding:
            if export_path is None:
                export_path = Path(self.config.encodeFilename)
            self.export_encoded_data(export_path)

    # compacts hypnodensity, possibly from mutliple models, into one Mx5 probability matrix.
    def get_hypnodensity(self):
        av = np.zeros(self.hypnodensity[0].shape)  # for example, 2144, 5)

        for i in range(len(self.hypnodensity)):
            av += self.hypnodensity[i]

        av = np.divide(av, len(self.hypnodensity))

        lights_on_mask = np.ones(av.shape[0]) == 1
        epoch_len: int = 15
        # Zero out portion with lights off
        lights_on_mask[self.config.get_lights_off_epoch(epoch_len=epoch_len):self.config.get_lights_on_epoch(epoch_len=epoch_len)] = False
        av[lights_on_mask, :] = np.nan

        bad_signal_events_1_sec = self.get_signal_quality_events()
        bad_signal_epoch_len = [self.epoch_rebase(x, 1, epoch_len).astype(np.uint32) for x in bad_signal_events_1_sec]
        # Consider also -->  bad_signal_epoch_len = self.config.sec2epoch(bad_signal_events_1_sec, epoch_len)

        # remove any sections identified with flatline
        for start_stop in bad_signal_epoch_len:
            # start_stop[1]+1 b/c the second value is not inclusive
            av[start_stop[0]:start_stop[1]+1, :] = np.nan
        return av

    # returns the number of hypnodensities available - this will correspond to the number of models used during the
    # inferencing step, which will be between 1 and 16.
    def get_num_hypnodensities(self):
        # return self._hypnodensity.hypnodensity.shape[0]
        return len(self.hypnodensity)

    # 0 is wake, 1 is stage-1, 2 is stage-2, 3 is stage 3/4, 5 is REM
    def get_hypnogram(self, epoch_len: int = 15):
        """
        :param epoch_len: Length of hypnogram epoch.  Can be 15 or 30.  Default is 15 s epochs
        :return: hypnogram vector codified as
         0 - wake
         1 - stage 1 sleep
         2 - stage 2 sleep
         3 - stage 3/4 sleep
         5 - rapid eye movement sleep
         7 - unscored (artifact or lights on)
        """
        hypno = self.get_hypnodensity()
        if epoch_len == 30:
            # The default is a 15 sec epoch, which is a segsize of 60 ...
            s = hypno.shape
            if s[0] % 2:
                # Add an extra row of zeros if we are short 15 s for a 30 s epoch
                hypno = np.append(hypno, np.tile(np.nan, [1, s[1]]), axis=0)
                # hypno = np.append(hypno, np.zeros(shape=(1, s[1]), dtype=hypno.dtype), axis=0)
            # Collapse
            hypno = hypno.reshape(-1, 2, hypno.shape[-1]).sum(1)

        # 0 is wake, 1 is stage-1, 2 is stage-2, 3 is stage 3/4, 4 is REM
        hypnogram = np.argmax(hypno, axis=1)

        # Change 4 to 5 to keep with the conventional REM indicator
        hypnogram[hypnogram == 4] = 5

        # Use '7' to identify unstaged.  This occurs where there are nan (not-a-number) values.  nan results when
        # there is more than 5 minutes of flat line data and also when the lights are on.
        hypnogram[np.isnan(hypno[:, 0])] = 7
        return hypnogram

    def encoding(self):

        def encode_data(x1, x2, dim, slide, fs):
            # Length of the first dimension and overlap of segments
            dim = int(fs * dim)
            slide = int(fs * slide)

            # Create 2D array of overlapping segments
            zero_vec = np.zeros(dim // 2)
            input2 = np.concatenate((zero_vec, x2, zero_vec))
            D1 = rolling_window_nodelay(x1, dim, slide)
            D2 = rolling_window_nodelay(input2, dim * 2, slide)
            # D1 = skimage.util.view_as_windows(x1, dim, slide).T
            # D2 = skimage.util.view_as_windows(input2, dim * 2, slide).T
            zero_mat = np.zeros((dim // 2, D1.shape[1]))
            D1 = np.concatenate([zero_mat, D1, zero_mat])

            keep_dims = D1.shape[1]
            D2 = D2[:, :keep_dims]
            D1 = D1[:, :keep_dims]
            # C = tf.signal.fftshift(nf.real(tf.signal.ifft2d(tf.signal.fft2d(D1.astype(np.complex128)) * np.conj(tf.signal.fft2d(D2.astype(np.complex128))))))

            # See: https: // www.tensorflow.org / api_docs / python / tf / nn / conv2d
            # a = np.array([[[[2], [1], [2]], [[1], [2], [3]], ]])
            # b = tf.nn.conv2d(a, a, strides=[1, 1, 1, 1], padding='VALID')
            # b = tf.raw_ops.Conv2D(input=a, filter=a, strides=[1, 1, 1, 1], padding='VALID')
            # Fast implementation of auto/cross-correlation
            C = fftshift(
                np.real(ifft(fft(D1, dim * 2 - 1, axis=0) * np.conj(fft(D2, dim * 2 - 1, axis=0)), axis=0)),
                axes=0).astype(dtype=np.float32)

            # Remove mirrored part
            C = C[dim // 2 - 1: - dim // 2]

            # Scale data with log modulus
            scale = np.log(np.max(np.abs(C) + 1, axis=0) / dim)
            C = C[..., :] / (np.amax(np.abs(C), axis=0) / scale)

            return C

        count = -1
        enc = []

        for c in self.channels_used:  # Central, Occipital, EOG-L, EOG-R, chin
            # append autocorrelations
            enc.append(encode_data(self.loaded_channels[c], self.loaded_channels[c], self.cc_size[c], 0.25, self.fs))

        # Append eog cross correlation
        enc.append(
            encode_data(self.loaded_channels['EOG-L'], self.loaded_channels['EOG-R'], self.cc_size['EOG-L'], 0.25,
                        self.fs))
        min_length = np.min([x.shape[1] for x in enc])
        enc = [v[:, :min_length] for v in enc]

        # Central, Occipital, EOG-L, EOG-R, EOG-L/R, chin
        enc = np.concatenate([enc[0], enc[1], enc[2], enc[3], enc[5], enc[4]], axis=0)
        self.encoded_data = enc

        # Adjust for lights off/on
        # fps = 4  # encoded data is sampled at 4 Hz, (i.e. a period of 0.25 s)
        # if isinstance(self.lights_off, int) and isinstance(self.lights_on, int):
        #     _encoded_data = self.encoded_data[:, fps * self.lights_off: fps * self.lights_on]
        # else:
        #     _encoded_data = self.encoded_data

    def loadEDF(self):
        if not self.edf:
            try:
                self.edf = pyedflib.EdfReader(self.edf_filename)
            except OSError as osErr:
                print("OSError:", "Loading", self.edf_filename)
                raise osErr

        for ch in self.channels:  # ['C3','C4','O1','O2','EOG-L','EOG-R','EMG','A1','A2']
            self.myprint('Loading', ch)
            if isinstance(self.channels_used[ch], int):

                self.loaded_channels[ch] = self.edf.readSignal(self.channels_used[ch])
                if self.edf.getPhysicalDimension(self.channels_used[ch]).lower() == 'mv':
                    self.myprint('mv')
                    self.loaded_channels[ch] *= 1e3
                elif self.edf.getPhysicalDimension(self.channels_used[ch]).lower() == 'v':
                    self.myprint('v')
                    self.loaded_channels[ch] *= 1e6

                fs = int(self.edf.samplefrequency(self.channels_used[ch]))
                # fs = Decimal(fs).quantize(Decimal('.0001'), rounding=ROUND_DOWN)
                print('fs', fs)

                self.resampling(ch, fs)
                print('Resampling done')

                # Trim excess
                self.trim(ch)

            else:
                print('channel[', ch, '] was empty (skipped)', sep='')
                del self.channels_used[ch]

    def trim(self, ch):
        # 30 represents the epoch length most often used in standard hypnogram scoring.
        rem = len(self.loaded_channels[ch]) % int(self.fs * 30)
        # Otherwise, if rem == 0, the following results in an empty array
        if rem > 0:
            self.loaded_channels[ch] = self.loaded_channels[ch][:-rem]

    def loadHeader(self):
        if not self.edf:
            print(self.edf_filename)
            self.edf = pyedflib.EdfReader(self.edf_filename)

        signal_labels = self.edf.getSignalLabels()
        return signal_labels

    def filtering(self):
        self.myprint('Filtering remaining signals')
        fs = self.fs

        Fh = signal.butter(5, self.fs_high / (fs / 2), btype='highpass', output='ba')
        Fl = signal.butter(5, self.fs_low / (fs / 2), btype='lowpass', output='ba')

        for ch, ch_idx in self.channels_used.items():
            # Fix for issue 9: https://github.com/Stanford-STAGES/stanford-stages/issues/9
            if isinstance(ch_idx, int):
                self.myprint('Filtering {}'.format(ch))
                # See https://dsp.stackexchange.com/questions/11466/differences-between-python-and-matlab-filtfilt-function regarding
                # discrepancies between zero-padding performed in MATLAB's filtfilt and scipy's.
                # In matlab's filtfilt, it is 3*(max(len(a), len(b)) - 1), and in scipy's filtfilt, it is 3*max(len(a), len(b)).
                # self.loaded_channels[ch] = signal.filtfilt(Fh[0], Fh[1], self.loaded_channels[ch])
                padlen = 3*(max(len(Fh[0]), len(Fh[1])) - 1)
                self.loaded_channels[ch] = signal.filtfilt(Fh[0], Fh[1], self.loaded_channels[ch], padtype='odd',
                                                           padlen=padlen)
                if fs > (2 * self.fs_low):
                    padlen = 3 * (max(len(Fl[0]), len(Fl[1])) - 1)
                    self.loaded_channels[ch] = signal.filtfilt(Fl[0], Fl[1], self.loaded_channels[ch], padtype='odd',
                                                               padlen=padlen).astype(dtype=np.float32)

    def resampling(self, ch, fs):
        self.myprint("original samplerate = ", fs)
        if fs == self.fs:
            self.myprint("resampling not required")
        else:
            self.myprint("resampling to ", self.fs)
            resample_fs_id = str(self.fs)
            original_fs_id = str(fs)
            has_valid_filter_specs = resample_fs_id in self.filter_specs and\
                                     original_fs_id in self.filter_specs[resample_fs_id]
            if has_valid_filter_specs:
                try:
                    self.loaded_channels[ch] = signal.upfirdn(
                        self.filter_specs[resample_fs_id][original_fs_id]["numerator"],
                        self.loaded_channels[ch],
                        self.filter_specs[resample_fs_id][original_fs_id]["up"],
                        self.filter_specs[resample_fs_id][original_fs_id]["down"],
                    )
                    if self.fs == 100:
                        if fs == 256 or fs == 512:  # Matlab creates a filtercascade which requires the 128 Hz filter be applied afterwards
                            self.loaded_channels[ch] = signal.upfirdn(
                                self.filter_specs[resample_fs_id]["128"]["numerator"],
                                self.loaded_channels[ch],
                                self.filter_specs[resample_fs_id]["128"]["up"],
                                self.filter_specs[resample_fs_id]["128"]["down"],
                            )
                except:
                    self.myprint('Warning: Exception caught during resampling.  Using automated resampling filter '
                                 'coefficients.')
                    self.loaded_channels[ch] = signal.resample_poly(self.loaded_channels[ch], self.fs, fs, axis=0,
                                                                    window=('kaiser', 5.0))

            else:
                self.loaded_channels[ch] = signal.resample_poly(self.loaded_channels[ch], self.fs, fs, axis=0,
                                                                window=('kaiser', 5.0))

    def psg_noise_level(self):
        # Only need to check noise levels when we have two central or occipital channels
        # which we should then compare for quality and take the best one.  We can test this
        # by first checking if there is a channel category 'C4' or 'O2'
        has_c4 = self.channels_used.get('C4') is not None
        has_o2 = self.channels_used.get('O2') is not None

        # Update for issue #6 - The original code did assumed presence of C4 or O2 meant presence of C3 and O1, which is
        # not valid.  Need to explicitly ensure we have both channels when checking noise.
        has_c3 = self.channels_used.get('C3') is not None
        has_o1 = self.channels_used.get('O1') is not None

        has_centrals = has_c3 and has_c4
        has_occipitals = has_o1 and has_o2

        if has_centrals or has_occipitals:
            # print(f'Loading noise file: {self.config.psg_noise_file_pathname}\n')
            noiseM = sio.loadmat(self.config.psg_noise_file_pathname, squeeze_me=True)['noiseM']
            meanV = noiseM['meanV'].item()  # 0 for Central,    idx_central = 0
            covM = noiseM['covM'].item()  # 1 for Occipital,  idx_occipital = 1

            if has_centrals:
                centrals_idx = 0
                unused_ch = self.get_loudest_channel(['C3', 'C4'], meanV[centrals_idx], covM[centrals_idx])
                if unused_ch == 'C3':
                    print('Selecting C4')
                else:
                    print('Selecting C3')
                del self.channels_used[unused_ch]

            if has_occipitals:
                occipitals_idx = 1
                unused_ch = self.get_loudest_channel(['O1', 'O2'], meanV[occipitals_idx], covM[occipitals_idx])
                if unused_ch == 'O1':
                    print('Selecting O2')
                else:
                    print('Selecting O1')
                del self.channels_used[unused_ch]

    def get_loudest_channel(self, channel_tags, mean_vec, cov_mat):
        noise = np.zeros(len(channel_tags))
        for [idx, ch] in enumerate(channel_tags):
            noise[idx] = self.channel_noise_level(ch, mean_vec, cov_mat)
        return channel_tags[np.argmax(noise)]

        # for ch in channelTags:
        #     noise = self.channel_noise_level(ch, meanV, covM)
        #     if noise >= loudest_noise:
        #         loudest_noise = noise
        #         loudest_ch = ch
        # return loudest_ch

    def channel_noise_level(self, channel_tag, mean_vec, cov_mat):
        hjorth = Hypnodensity.extract_hjorth(self.loaded_channels[channel_tag], self.fs)
        noise_vec = np.zeros(hjorth.shape[1])
        for k in range(len(noise_vec)):
            M = hjorth[:, k][:, np.newaxis]
            x = M - mean_vec[:, np.newaxis]
            sigma = np.linalg.inv(cov_mat)
            noise_vec[k] = np.sqrt(np.dot(np.dot(np.transpose(x), sigma), x))
        return np.nanmean(noise_vec)  # ignore nan's which may pop up from hjorth calculation

    def score_data(self):
        self.hypnodensity = list()

        # bad_blocks = self.get_inf_nan_encoded_blocks()
        # self.encoded_data[:, bad_blocks] = 0
        inf_nan_indices = self.get_inf_nan_encoding_indices()
        self.encoded_data[inf_nan_indices] = 0

        # identify bad signal/data: flat line, excessive noise
        #self.identify_flatline()
        # flatline_15_sec = [self.epoch_rebase(x, 0.25, 15).astype(np.uint32) for x in self.flatline]
        # flatline_15_sec = []
        for l in self.config.models_used:
            hyp = self.run_data(self.encoded_data, l, self.config.hypnodensity_model_root_path)
            hyp = softmax(hyp)
            self.hypnodensity.append(hyp)

    # Returns a np.2darray of the start and stop times (elapsed times in seconds from start of the psg)
    # of events annotated as bad quality.  Data in these locations will be replaced with nan values.
    def get_signal_quality_events(self):
        # See if there is a file with the same name as the
        quality_control_events = []

        data_quality_file = Path(self.config.filename["bad_data"])
        # self.load_signal_quality_events(bP)

        if data_quality_file.exists():
            a = read_csv(data_quality_file, header=0, names=['start_sec', 'duration_sec', 'channel_label'])
            b = np.array(a[['start_sec', 'duration_sec']])
            b[:, 1] = np.sum(b, axis=1)
            quality_control_events = b
        return quality_control_events

    def get_inf_nan_encoding_indices(self):
        return np.logical_or(np.isnan(self.encoded_data), np.isinf(self.encoded_data))

    def get_inf_nan_encoded_blocks(self):
        bad_values = np.any(self.get_inf_nan_encoding_indices(), axis=0)
        bad_indices = np.where(bad_values)
        return bad_indices

    def identify_flatline(self):
        self.flatline = []
        bad_values = np.any(np.logical_or(np.isnan(self.encoded_data), np.isinf(self.encoded_data)), axis=0)
        # bad_indices = np.where(bad_values)
        # Another approach is to string together 2.5 segments from each channel.
        _flatline = np.tile(False, self.encoded_data.shape[1])
        for offset in self._encoded_data_channel_offsets.values():
            channel_slice = self.encoded_data[offset]
            _flatline = np.logical_or(_flatline, np.logical_or(np.isnan(channel_slice), np.isinf(channel_slice)))

        if any(bad_values):
            true_count = 0
            count_threshold = 30 / 2.5  # reject 30s or more of flatline.  encoded_data values come in 2.5s segments.
            for index, value in enumerate(bad_values):
                if value:
                    true_count = true_count + 1
                elif true_count > 0:
                    # This is the case where we have had a run (1 or more) bad values, have now had a good value, and
                    # now need to do book-keeping for the bad_value indices, record them if they pass the run count threshold.
                    if true_count >= count_threshold:
                        # index-1 because we don't include the current index which is false.
                        # really [(index-1)-true_count+1, index - 1] @hyatt
                        self.flatline.append(np.array([index-true_count, index-1]))
                    true_count = 0
            # Handle case of finishing with flat line data
            if true_count >= count_threshold:
                # NO -1 because we will include the most recent index since, which was positive.
                self.flatline.append(np.array([index - true_count, index]))

    def myprint(self, string, *args):
        if self.config.verbose:
            myprint(string, *args)

    @staticmethod
    def epoch_rebase(e, source_base_sec, new_base_sec):
        '''
        Rebase the epoch index e using the new base provided.
        :param e: The epoch index to rebase.
        :param source_base_sec: The number of seconds used for the base of e
        :param new_base_sec: The new base to rebase e to.
        :return: e with base of new_base_sec
        '''
        #e_1_sec_base = e*source_base_sec
        #e_new_base_sec = e_1_sec_base // new_base_sec
        #return e_new_base_sec
        return (e*source_base_sec) // new_base_sec

    # Use 5 minute sliding window.
    @staticmethod
    def extract_hjorth(x, fs, dim=5 * 60, slide=5 * 60):

        # Length of first dimension
        dim = dim * fs

        # Overlap of segments in samples
        slide = slide * fs

        # Creates 2D array of overlapping segments
        # Ref issue #24
        D = rolling_window_nodelay(x, dim, slide)
        D = np.delete(D, -1, axis=-1)

        # Extract Hjorth params for each segment
        dD = np.diff(D, 1, axis=0)
        ddD = np.diff(dD, 1, axis=0)
        mD2 = np.mean(D ** 2, axis=0)
        mdD2 = np.mean(dD ** 2, axis=0)
        mddD2 = np.mean(ddD ** 2, axis=0)

        top = np.sqrt(np.divide(mddD2, mdD2))

        # Mobility
        # mobil = self.mob(B)
        # def mob(b):
        #     diff = np.diff(b, axis=0)
        #     var = np.var(diff, axis=0)
        #     return np.sqrt(np.divide(var, np.var(b, axis=0)))

        mobility = np.sqrt(np.divide(mdD2, mD2))
        activity = mD2
        complexity = np.divide(top, mobility)

        hjorth = np.array([activity, complexity, mobility])
        hjorth = np.log(hjorth + np.finfo(float).eps)
        return hjorth

    @staticmethod
    def run_data(dat, model, root_model_path):
        ac_config = ACConfig(model_name=model, is_training=False, root_model_dir=root_model_path)
        hyp = Hypnodensity.run(dat, ac_config)
        return hyp

    @staticmethod
    def segment(dat, ac_config):
        # Get integer value for segment size using //
        n_seg = dat.shape[1] // ac_config.segsize

        dat = np.expand_dims(dat[:, :n_seg * ac_config.segsize], 0)

        num_batches = np.int(
            np.ceil(np.divide(dat.shape[2], (ac_config.eval_nseg_atonce * ac_config.segsize), dtype='float')))

        # print(f'\n---------------\ndat.shape[2] = {dat.shape[2]}\nnum_batches = {num_batches}\n--------------------\n')

        # n_extra = np.int(np.ceil(num_batches * ac_config.eval_nseg_atonce * ac_config.segsize) % dat.shape[2])
        n_extra = num_batches * ac_config.eval_nseg_atonce * ac_config.segsize - dat.shape[2]

        # fill remaining (n_extra) values with the mean value of each column
        meanF = np.mean(np.mean(dat, 2), 0) * np.ones([1, n_extra, dat.shape[1]])

        dat = np.transpose(dat, [0, 2, 1])
        dat = np.concatenate([dat, meanF], 1)

        prediction = np.zeros([num_batches * ac_config.eval_nseg_atonce, 5])

        return dat, n_extra, prediction, num_batches

    @staticmethod
    def run(dat, ac_config):

        with tf.compat.v1.Graph().as_default() as g:
            m = SCModel(ac_config)
            s = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

            print("AC config hypnodensity path", ac_config.hypnodensity_model_dir)
            # config = tf.compat.v1.ConfigProto(log_device_placement=False, device_count={'GPU': 0}) # For cpu only operations
            config = tf.compat.v1.ConfigProto(log_device_placement=False)
            # config = tf.ConfigProto()
            # config = tf.ConfigProto(log_device_placement=True)  # Setting log_device_placement=True gives way too much output.
            # config.gpu_options.allow_growth = True  # See also: https://www.tensorflow.org/guide/using_gpu
            # config.gpu_options.per_process_gpu_memory_fraction = 1.0
            with tf.compat.v1.Session(config=config) as session:
                ckpt = tf.compat.v1.train.get_checkpoint_state(ac_config.hypnodensity_model_dir)

                if ckpt is None:
                    raise StanfordStagesError(f"Hypnodensity model directory is empty or does not exist ('{ac_config.hypnodensity_model_dir}')")

                s.restore(session, ckpt.model_checkpoint_path)

                state = np.zeros([1, ac_config.num_hidden * 2])
                # state = m.initial_state

                dat, Nextra, prediction, num_batches = Hypnodensity.segment(dat, ac_config)
                for i in range(num_batches):
                    x = dat[:, i * ac_config.eval_nseg_atonce * ac_config.segsize:(i + 1) *
                                                                                  ac_config.eval_nseg_atonce * ac_config.segsize,
                        :]

                    est, state = session.run([m.logits, m.final_state], feed_dict={
                        m.features: x,
                        m.targets: np.ones([ac_config.eval_nseg_atonce * ac_config.segsize, 5]),
                        m.mask: np.ones(ac_config.eval_nseg_atonce * ac_config.segsize),
                        m.batch_size: np.ones([1]),
                        m.initial_state: state
                    })

                    prediction[i * ac_config.eval_nseg_atonce:(i + 1) * ac_config.eval_nseg_atonce, :] = est

                prediction = prediction[:-int(Nextra / ac_config.segsize), :]

                return prediction

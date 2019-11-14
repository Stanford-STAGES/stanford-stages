# -*- coding: utf-8 -*-
"""
@author: jens
@modifier: informaton

inf_narco_biomarker --> inf_narco_app
"""
import json  # for command line interface input and output.
import os
import sys
import warnings
# from asyncore import file_dispatcher
from datetime import datetime

import gpflow as gpf
# For hypnodensity plotting ...
import matplotlib.pyplot as plt
import numpy as np
# import pdb
import tensorflow as tf
from matplotlib.patches import Polygon

from inf_config import AppConfig  # for AppConfig() <-- narco_biomarker(), [previously]
from inf_hypnodensity import Hypnodensity  # from inf_extract_features import ExtractFeatures --> moved to
                                           # inf_hypnodensity.py

# for auditing code speed.
from pathlib import Path
import time

warnings.simplefilter('ignore', FutureWarning)  # warnings.filterwarnings("ignore")

DEBUG_MODE = False
STANDARD_EPOCH_SEC = 30
DEFAULT_SECONDS_PER_EPOCH = 30
DEFAULT_MINUTES_PER_EPOCH = 0.5  # 30/60 or DEFAULT_SECONDS_PER_EPOCH/60;

# The algorithm produces values between −1 and 1, with 1 indicating a high probability of narcolepsy.
# The cut-off threshold between narcolepsy type 1 and “other“ is set at −0.03.
# Ref: https://www.nature.com/articles/s41467-018-07229-3
NARCOLEPSY_PREDICTION_CUTOFF = -0.03
DIAGNOSIS = ["Other", "Narcolepsy type 1"]


def main(edf_filename,
         config_input):  # configInput is object with additional settings.   'channel_indices', 'lightsOff','lightsOn'

    # Application settings
    app_config = AppConfig()
    app_config.edf_path = edf_filename

    channel_categories = {
        'central': 'C3',
        'central3': 'C3',
        'central4': 'C4',
        'centrals': ('C3', 'C4'),
        'occipital': 'O1',
        'occipital1': 'O1',
        'occipital2': 'O2',
        'occipitals': ('O1', 'O2'),
        'eog_l': 'EOG-L',
        'eog_r': 'EOG-R',
        'eogs': ('EOG-L', 'EOG-R'),
        'chin_emg': 'EMG'
    }

    for channel_category, channel_index in config_input["channel_indices"].items():
        channel_label = channel_categories.get(channel_category, None)
        if channel_label is not None:
            if type(channel_index) is list or type(channel_index) is tuple:  # ==type(tuple):
                for i in range(len(channel_index)):
                    app_config.channels_used[channel_label[i]] = channel_index[i]
            else:
                app_config.channels_used[channel_label] = channel_index

    app_config.lightsOff = config_input.get('lightsOff', [])
    app_config.lightsOn = config_input.get('lightsOn', [])
    app_config.audit.update(config_input.get('audit',{}))

    hyp = {'show': {}, 'save': {}, 'filename': {}}
    hyp['show']['plot'] = False
    hyp['show']['hypnogram'] = False
    hyp['show']['hypnodensity'] = False
    hyp['show']['diagnosis'] = True

    hyp['save']['plot'] = True
    hyp['save']['hypnogram'] = True
    hyp['save']['hypnodensity'] = True
    hyp['save']['diagnosis'] = True
    hyp['save']['encoding'] = True

    hyp['filename']['plot'] = change_file_extension(edf_filename, '.hypnodensity.png')
    hyp['filename']['pkl_hypnodensity'] = change_file_extension(edf_filename, '.hypnodensity.pkl')
    hyp['filename']['h5_hypnodensity'] = change_file_extension(edf_filename, '.hypnodensity.h5')
    hyp['filename']['hypnodensity'] = change_file_extension(edf_filename, '.hypnodensity.txt')
    hyp['filename']['hypnogram'] = change_file_extension(edf_filename, '.hypnogram.txt')
    hyp['filename']['diagnosis'] = change_file_extension(edf_filename, '.diagnosis.txt')
    hyp['filename']['encoding'] = change_file_extension(edf_filename, '.h5')
    hyp['filename']['pkl_encoding'] = None
    hyp['filename']['h5_encoding'] = None
    hyp['filename']['audit'] = None

    # hyp['filename']['encoding'] = change_file_extension(edf_filename, '.h5')

    hyp['save'].update(config_input.get('save', {}))
    hyp['show'].update(config_input.get('show', {}))
    hyp['filename'].update(config_input.get('filename', {}))

    hypno_config = hyp

    # Auditing filename is used to identify if audit should be done (None: False) and if so, which file to send audit
    # data to (filename: True)
    app_config.filename = hyp['filename']
    # app_config.auditFilename = hyp['filename']['audit']
    # app_config.h5_encoding = hyp['filename']['h5_encoding']
    # app_config.pkl_encoding = hyp['filename']['pkl_encoding']

    app_config.saveEncoding = hyp['save']['encoding']
    app_config.saveHypnodensity = hyp['save']['hypnodensity']
    app_config.encodeFilename = hyp['filename']['encoding']
    app_config.encodeOnly = not (hyp['show']['hypnogram'] or hyp['show']['hypnodensity'] or hyp['save']['hypnogram'] or
                                 hyp['save']['hypnodensity'] or hyp['show']['diagnosis'] or hyp['save']['diagnosis'] or
                                 hyp['show']['plot'] or hyp['save']['plot'])

    narco_app = NarcoApp(app_config)
    # narcoApp.eval_all()
    narco_app.eval_hypnodensity()

    if narco_app.config.audit.get('diagnosis', False):
        narco_app.audit(narco_app.eval_narcolepsy, 'Diagnosing...')

    if hypno_config['show']['hypnogram']:
        print("Hypnogram:")
        hypnogram = narco_app.get_hypnogram()
        np.set_printoptions(threshold=10000, linewidth=150)  # use linewidth = 2 to output as a single column
        print(hypnogram)

    # This is the text portion
    if hypno_config['save']['hypnogram']:
        narco_app.save_hypnogram(filename=hypno_config['filename']['hypnogram'])

    if hypno_config['show']['hypnogram']:
        print("Hypnodensity:")
        hypnodensity = narco_app.get_hypnodensity()
        np.set_printoptions(threshold=10000*5, linewidth=150)
        print(hypnodensity)

    if hypno_config['save']['hypnodensity']:
        narco_app.save_hypnodensity(filename=hypno_config['filename']['hypnodensity'])

    if hypno_config['show']['diagnosis']:
        print(narco_app.get_diagnosis())

    if hypno_config['save']['diagnosis']:
        narco_app.save_diagnosis(filename=hypno_config['filename']['diagnosis'])

    if not app_config.encodeOnly:
        render_hypnodensity(narco_app.get_hypnodensity(), show_plot=hypno_config['show']['plot'],
                            save_plot=hypno_config['save']['plot'], filename=hypno_config['filename']['plot'])


def change_file_extension(fullname, new_extension):
    basename, _ = os.path.splitext(fullname)
    return basename + new_extension


def render_hypnodensity(hypnodensity, show_plot=False, save_plot=False, filename='tmp.png'):
    fig, ax = plt.subplots(figsize=[11, 5])
    av = np.cumsum(hypnodensity, axis=1)
    c = [[0.90, 0.19, 0.87],  # pink
         [0.2, 0.89, 0.93],   # aqua/turquoise
         [0.22, 0.44, 0.73],  # blue
         [0.34, 0.70, 0.39]]  # green

    for i in range(4):
        xy = np.zeros([av.shape[0] * 2, 2])
        xy[:av.shape[0], 0] = np.arange(av.shape[0])
        xy[av.shape[0]:, 0] = np.flip(np.arange(av.shape[0]), axis=0)

        xy[:av.shape[0], 1] = av[:, i]
        xy[av.shape[0]:, 1] = np.flip(av[:, i + 1], axis=0)

        poly = Polygon(xy, facecolor=c[i], edgecolor=None)
        ax.add_patch(poly)

    plt.xlim([0, av.shape[0]])
    # fig.savefig('test.png')
    if save_plot:
        fig.savefig(filename)
        # plt.savefig(fileName)

    if show_plot:
        print("Showing hypnodensity - close figure to continue.")
        plt.show()


class NarcoApp(object):

    def __init__(self, app_config):

        # appConfig is an instance of AppConfig class, defined in inf_config.py
        self.config = app_config
        self.edf_path = app_config.edf_path  # full filename of an .EDF to use for header information.  A template .edf

        self.Hypnodensity = Hypnodensity(app_config)
        self.models_used = app_config.models_used
        self.narcolepsy_probability = []

    def get_diagnosis(self):
        prediction = self.narcolepsy_probability
        if not prediction:
            prediction = self.get_narco_prediction()
        return "Score: %0.4f\nDiagnosis: %s" % \
               (prediction[0], DIAGNOSIS[int(prediction >= NARCOLEPSY_PREDICTION_CUTOFF)])

    def get_hypnodensity(self):
        return self.Hypnodensity.get_hypnodensity()

    def get_hypnogram(self):
        return self.Hypnodensity.get_hypnogram()

    def save_diagnosis(self, filename=''):
        if filename == '':
            filename = change_file_extension(self.edf_path, '.diagnosis.txt')
        with open(filename, "w") as textFile:
            print(self.get_diagnosis(), file=textFile)

    def save_hypnodensity(self, filename=''):
        if filename == '':
            filename = change_file_extension(self.edf_path, '.hypnodensity.txt')
        hypno = self.get_hypnodensity()
        np.savetxt(filename, hypno, delimiter=",")

    def save_hypnogram(self, filename=''):
        if filename == '':
            filename = change_file_extension(self.edf_path, '.hypnogram.txt')

        hypno = self.get_hypnogram()
        np.savetxt(filename, hypno, delimiter=",", fmt='%i')

    def audit(self, method_to_audit, audit_label, *args):
        start_time = time.time()
        method_to_audit(*args)
        elapsed_time = time.time() - start_time
        with Path(self.config.filename['audit']).open('a') as fp:
            audit_str = f', {audit_label}: {elapsed_time:0.3f} s'
            fp.write(audit_str)

    def get_narco_gpmodels(self):
        return self.models_used

    def get_hypnodensity_features(self, model_name, idx):
        return self.Hypnodensity.get_features(model_name, idx)

    def get_narco_prediction(self):  # ,current_subset, num_subjects, num_models, num_folds):
        # Initialize dummy variables
        num_subjects = 1
        gpmodels = self.get_narco_gpmodels()
        num_models = len(gpmodels)
        num_folds = self.config.narco_prediction_num_folds

        mean_pred = np.zeros([num_subjects, num_models, num_folds])
        var_pred = np.zeros([num_subjects, num_models, num_folds])

        scales = self.config.narco_prediction_scales
        gpmodels_base_path = self.config.narco_classifier_path

        # config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        # config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True,))
        # config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True, ),
        #                                   log_device_placement=True,)

        # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

        # tf.debugging.set_log_device_placement(True)

        #  with tf.compat.v1.device('/GPU:0') as asif:
        # m = gpf.saver.Saver().load(gp_model_filename)  # Allocates GPU memory ...
        #mean_pred[:, idx, k, np.newaxis], var_pred[:, idx, k, np.newaxis] = m.predict_y(x)

        config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True),
                                          log_device_placement=False, )
        for idx, gpmodel in enumerate(gpmodels):
            print('{} | Predicting using: {}'.format(datetime.now(), gpmodel))

            x = self.get_hypnodensity_features(gpmodel, idx)

            for k in range(num_folds):
                # print('{} | Loading and predicting using {}'.format(datetime.now(), os.path.join(
                # gpmodels_base_path, gpmodel, gpmodel + '_fold{:02}.gpm'.format(k+1))))
                gp_model_filename = os.path.join(gpmodels_base_path, gpmodel,
                                                 gpmodel + '_fold{:02}.gpm'.format(k + 1))
                if not os.path.isfile(gp_model_filename):
                    print(f'MISSING Model: {gp_model_filename}\n')
                    continue
                else:
                    with tf.compat.v1.Graph().as_default() as graph:
                        # config = tf.compat.v1.ConfigProto(
                        #     gpu_options=tf.compat.v1.GPUOptions(allow_growth=True, visible_device_list='0'),
                        #     log_device_placement=False, device_count={'GPU': 1, 'CPU': 16})
                        #config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=False),
                        #                                  log_device_placement=False, )

                        with tf.compat.v1.Session(
                                config=config).as_default() as session:  # little gpu allocation and cuda usage
                            m = gpf.saver.Saver().load(gp_model_filename)  # Allocates GPU resources
                            # mean_pred[:, idx, k, np.newaxis], var_pred[:, idx, k, np.newaxis] = m.predict_y(x)
                            mean_pred[:, idx, k, np.newaxis], _ = m.predict_y(x)

        self.narcolepsy_probability = np.sum(np.multiply(np.mean(mean_pred, axis=2), scales), axis=1) / np.sum(scales)
        print(self.narcolepsy_probability[0])
        return self.narcolepsy_probability

    def eval_hypnodensity(self):
        self.Hypnodensity.evaluate()

    def eval_narcolepsy(self):
        self.get_narco_prediction()

    def eval_all(self):
        self.eval_hypnodensity()
        self.eval_narcolepsy()


if __name__ == '__main__':
    outputFormat = 'json'

    if sys.argv[1:]:  # if there are at least three arguments (two beyond [0])

        edfFile = sys.argv[1]

        # For hard coding/bypassing json input argument, uncomment the following: jsonObj = json.loads('{
        # "channel_indices":{"centrals":[3,4],"occipitals":[5,6],"eog_l":7,"eog_r":8,"chin_emg":9},
        # "show":{"plot":false,"hypnodensity":false,"hypnogram":false}, "save":{"plot":false,"hypnodensity":true,
        # "hypnogram":true}}')
        jsonObj = json.loads(sys.argv[2])
        try:
            main(edfFile, jsonObj)
        except OSError as oserr:
            print("OSError:", oserr)
    else:
        print(sys.argv[0], 'requires two arguments when run as a script')

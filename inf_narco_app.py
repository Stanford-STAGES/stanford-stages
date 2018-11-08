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
from datetime import datetime

warnings.simplefilter('ignore', FutureWarning)  # warnings.filterwarnings("ignore")

# for getting predictions
import scipy.io as sio

import tensorflow as tf
# import gpflow as gpf
import random

# For hypnodensity plotting ...
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from inf_hypnodensity import \
    Hypnodensity  # from inf_extract_features import ExtractFeatures --> moved to inf_hypnodensity.py
from inf_config import AppConfig  # for AppConfig() <-- narco_biomarker(), [previously]

DEBUG_MODE = False
STANDARD_EPOCH_SEC = 30
DEFAULT_SECONDS_PER_EPOCH = 30
DEFAULT_MINUTES_PER_EPOCH = 0.5  # 30/60 or DEFAULT_SECONDS_PER_EPOCH/60;


def main(edfFilename,
         configInput):  # configInput is object with additional settings.   'channel_indices', 'lightsOff','lightsOn'

    # Application settings
    appConfig = AppConfig()
    appConfig.edf_path = edfFilename;

    channel_categories = {
        'central': 'C3',
        'central3': 'C3',
        'central4': 'C4',
        'centrals': ('C3','C4'),
        'occipital': 'O1',
        'occipital1': 'O1',
        'occipital2': 'O2',
        'occipitals': ('O1','O2'),
        'eog_l': 'EOG-L',
        'eog_r': 'EOG-R',
        'eogs': ('EOG-L','EOG-R'),
        'chin_emg': 'EMG'
    }

    # configInput.channel_labels['central','occipital','eog_l','eog_r','chin_emg']
    # configInput.channel_indices
    for channel_category, channel_index in configInput["channel_indices"].items():
        channel_label = channel_categories.get(channel_category, None)
        if channel_label is not None:
            if type(channel_index) is list or type(channel_index) is tuple:   # ==type(tuple):
                for i in range(len(channel_index)):
                    appConfig.channels_used[channel_label[i]] = channel_index[i]
            else:
                appConfig.channels_used[channel_label] = channel_index


    appConfig.lightsOff = configInput.get('lightsOff',[]) #33)
    appConfig.lightsOn = configInput.get('lightsOn',[]) #1152)
    #appConfig.showPlot = configInput.get('hypnodensity.showplot',False)
    hyp = {}
    hyp['showplot'] = False
    hyp['saveplot'] = False
    hyp['showtext'] = False
    hyp['savetext'] = False
    hyp['savename'] = edfFilename
    hyp.update(configInput.get('hypnodensity', {}))
    hypnoConfig = hyp

    narcoApp = NarcoApp(appConfig)

    # narcoApp.eval_all()
    narcoApp.eval_hypnodensity()

    print('AppConfig.showPlot is ', hypnoConfig["showplot"])
    print('AppConfig.saveplot is ', hypnoConfig["saveplot"])

    renderHypnodensity(narcoApp.get_hypnodensity(), showPlot=hypnoConfig['showplot'],
                       savePlot=hypnoConfig['saveplot'], fileName=hypnoConfig['savename'])
    prediction = narcoApp.get_narco_prediction()

    print(prediction)


def renderHypnodensity(hypnodensity, showPlot=False, savePlot=False, fileName='tmp.png'):
    fig, ax = plt.subplots(figsize=[11, 5])
    av = np.cumsum(hypnodensity, axis=1)
    C = [[0.90, 0.19, 0.87],
         [0.2, 0.89, 0.93],
         [0.22, 0.44, 0.73],
         [0.34, 0.70, 0.39]]

    for i in range(4):
        xy = np.zeros([av.shape[0] * 2, 2])
        xy[:av.shape[0], 0] = np.arange(av.shape[0])
        xy[av.shape[0]:, 0] = np.flip(np.arange(av.shape[0]), axis=0)

        xy[:av.shape[0], 1] = av[:, i]
        xy[av.shape[0]:, 1] = np.flip(av[:, i + 1], axis=0)

        poly = Polygon(xy, facecolor=C[i], edgecolor=None)
        ax.add_patch(poly)

    plt.xlim([0, av.shape[0]])
    # fig.savefig('test.png')
    if savePlot:
        fig.savefig(fileName)
        # plt.savefig(fileName)

    if showPlot:
        print("Showing hypnodensity")
        plt.show()

    pdb.set_trace()

class NarcoApp(object):

    def __init__(self, appConfig):

        # appConfig is an instance of AppConfig class, defined in inf_config.py
        self.config = appConfig
        self.edf_path = appConfig.edf_path  # full filename of an .EDF to use for header information.  A template .edf

        self.Hypnodensity = Hypnodensity(appConfig)

        self.models_used = appConfig.models_used

        self.edfeatureInd = []
        self.narco_features = []
        self.narcolepsy_probability = []
        # self.extract_features = ExtractFeatures(appConfig)  <-- now in Hypnodensity

    def get_hypnodensity(self):
        return self.Hypnodensity.get_hypnodensity()

    def get_narco_gpmodels(self):

        return self.models_used

    def get_hypnodensity_features(self, modelName, idx):
        return self.Hypnodensity.get_features(modelName, idx)

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

        for idx, gpmodel in enumerate(gpmodels):
            print('{} | Predicting using: {}'.format(datetime.now(), gpmodel))

            if False:
                # Load the data
                mat_file_path = os.path.join(current_subset, gpmodel + '_testFeatures.mat')
                X = sio.loadmat(mat_file_path)['features'].T
                selected_features = self.config.narco_prediction_selected_features
                X = X[:, selected_features]
            else:
                X = self.get_hypnodensity_features(gpmodel, idx)

            for k in range(num_folds):
                #         print('{} | Loading and predicting using {}'.format(datetime.now(), os.path.join(gpmodels_base_path, gpmodel, gpmodel + '_fold{:02}.gpm'.format(k+1))))
                with tf.Graph().as_default() as graph:
                    with tf.Session():
                        m = gpf.saver.Saver().load(
                            os.path.join(gpmodels_base_path, gpmodel, gpmodel + '_fold{:02}.gpm'.format(k + 1)))
                        mean_pred[:, idx, k, np.newaxis], var_pred[:, idx, k, np.newaxis] = m.predict_y(X)

        self.narcolepsy_probability = np.sum(np.multiply(np.mean(mean_pred, axis=2), scales), axis=1) / np.sum(scales)
        return self.narcolepsy_probability

    def plotHypnodensity(self):
        self.Hypnodensity.renderHynodenisty(option="plot")

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

        if sys.argv[1] == 'demo':

            # print results
            results = {}

            randomTest = sys.argv[2] == "random"
            if randomTest:
                random.seed()
                ml_pred = random.triangular(0.0, 0.001, 0.95)
            else:
                ml_pred = 0.001

            ml_threshold = 0.5
            ml_result = ml_pred >= ml_threshold

            edfObj = edfClass(filename="demo.edf");
            results["machine_learning"] = (ml_result, ml_pred, ml_threshold)
            results["reml"] = edfObj.testREMLatency(randomTest)
            results["soremp"] = edfObj.testSOREMPs(randomTest)
            results["nrem"] = edfObj.testNREMFragmentation(randomTest)
            results["pwakes"] = edfObj.testPWakes(randomTest)

            results["hypnogram"] = [1, 2, 3, 4, 5, 0, 1, 2, 3]
            results["hypnodensity_image"] = 'tmp.png'
            print(json.dumps(results))

        else:

            edfFile = sys.argv[1]
            # print(sys.argv[2]);
            jsonObj = json.loads('{"channel_indices":{"central":3,"occipital":0,"eog_l":0,"eog_r":0,"chin_emg":0}, "hypnodensity":{"showplot":true}}')
            # jsonObj = json.loads(sys.argv[2])
            try:
                main(edfFile, jsonObj)
            except OSError as oserr:
                print("OSError:", oserr)


    else:
        print(sys.argv[0], 'requires two arguments when run as a script')

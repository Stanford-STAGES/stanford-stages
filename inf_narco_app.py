# -*- coding: utf-8 -*-
"""
@author: jens
@modifier: informaton

inf_narco_biomarker --> inf_narco_app
"""
import json  # for command line interface input and output.
import os, sys, warnings
from pathlib import Path
import logging
# from asyncore import file_dispatcher
# from typing import Any, Union

import gpflow as gpf
# For hypnodensity plotting ...
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from inf_config import AppConfig  # for AppConfig() <-- narco_biomarker(), [previously]
from inf_hypnodensity import Hypnodensity  # from inf_extract_features import ExtractFeatures --> moved to
                                           # inf_hypnodensity.py

from inf_narco_model import NarcoModel
from inf_tools import StanfordStagesError

# for auditing code speed.
import time


warnings.simplefilter('ignore', FutureWarning)  # warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
DEBUG_MODE = False
STANDARD_EPOCH_SEC = 30
DEFAULT_SECONDS_PER_EPOCH = 30
DEFAULT_MINUTES_PER_EPOCH = 0.5  # 30/60 or DEFAULT_SECONDS_PER_EPOCH/60;

# The algorithm produces values between −1 and 1, with 1 indicating a high probability of narcolepsy.
# The cut-off threshold between narcolepsy type 1 and “other“ is set at −0.03.
# Ref: https://www.nature.com/articles/s41467-018-07229-3
NARCOLEPSY_PREDICTION_CUTOFF = -0.03
DIAGNOSIS = ["Other", "Narcolepsy type 1"]


def main(edf_filename: str = None,
         config_input: {} = None, config_filename: str = None):  # configInput is object with additional settings.   'channel_indices', 'lightsOff','lightsOn'

    err_msg = ''
    if edf_filename is None:
        err_msg += f"main() requires an edf_filename to run\n"
    if config_input is None and config_filename is None:
        err_msg += "main() requires a configuration dictionary or filename as input in order to run\n"
    if err_msg != '':
        raise StanfordStagesError(err_msg, edf_filename)

    # Application settings
    app_config = AppConfig()
    app_config.edf_filename = edf_filename

    if config_input is None:
        config_input = {}

    # Update config_input with anything found in the json file that does not exist as a key value pair in the
    # config_input dictionary

    if config_filename is not None:
        if not Path(config_filename).is_file():
            raise StanfordStagesError(f"config_filename ({config_filename}) does not exist", edf_filename)
        else:
            with open(config_filename, 'r') as fid:
                json_dict = json.load(fid)
            for key, value in json_dict.items():
                if key not in config_input:
                    config_input[key] = value

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
    edf_file: Path = Path(edf_filename)

    output_path = Path(config_input.get("output_path", edf_file.parent))

    inf_config_dict = config_input.get("inf_config", None)
    if isinstance(inf_config_dict, dict):
        keys_to_check = list(vars(app_config))
        # Add these properties
        properties = ['lights_off', 'lights_on']
        for prop in properties:
            if prop not in keys_to_check:
                keys_to_check.append(prop)
        # if 'lights_off' not in keys_to_check:
        #     keys_to_check.append('lights_off')
        # if 'lights_on' not in keys_to_check:
        #     keys_to_check.append('lights_on')

        for key in keys_to_check:
            if key in inf_config_dict and key != 'channels_used':
                value = inf_config_dict[key]
                # Tried to handle this a property and setter, but ran into issues with setattr and _narco_prediction_selected_features
                if key.lower() == 'narco_prediction_selected_features':
                    app_config.set_narco_feature_selection(value)
                elif isinstance(value, list) and not len(value):
                    continue
                else:
                    setattr(app_config, key, value)
        if app_config.narco_prediction_selected_features is None:
            print('No narcolepsy feature set.')
        else:
            print('Number of selected features is:', len(app_config.narco_prediction_selected_features))
    else:
        inf_config_dict = {}

    # Bypassing the edf check is useful in cases where the edf has already been processed to an .h5 encoding file
    # or hypnodensity file and the edf is no longer necessary (and perhaps removed for storage reasons)
    bypass_edf_check = config_input.get('bypass_edf_check', False)
    if not bypass_edf_check:
        for channel_category, channel_index in config_input["channel_indices"].items():
            channel_label = channel_categories.get(channel_category, None)
            if channel_label is not None:
                if type(channel_index) is list or type(channel_index) is tuple:  # ==type(tuple):
                    for i in range(len(channel_index)):
                        app_config.channels_used[channel_label[i]] = channel_index[i]
                else:
                    app_config.channels_used[channel_label] = channel_index


    # app_config.lightsOff = config_input.get('lightsOff', [])
    # app_config.lightsOn = config_input.get('lightsOn', [])
    # app_config.audit.update(config_input.get('audit',{}))

    # These are updated below based on the config_input
    hyp = {'show': {}, 'save': {}, 'filename': {}}
    hyp['show']['plot'] = False
    hyp['show']['hypnogram'] = False
    hyp['show']['hypnogram_30_sec'] = False
    hyp['show']['hypnodensity'] = False
    hyp['show']['diagnosis'] = False

    hyp['save']['plot'] = True
    hyp['save']['hypnogram'] = True
    hyp['save']['hypnogram_30_sec'] = True
    hyp['save']['hypnodensity'] = None
    hyp['save']['hypnodensity_txt'] = True
    hyp['save']['hypnodensity_pkl'] = True
    hyp['save']['hypnodensity_h5'] = True

    hyp['save']['diagnosis'] = False
    hyp['save']['encoding'] = True
    hyp['save']['features'] = None
    hyp['save']['features_pkl'] = False
    hyp['save']['features_h5'] = False

    hyp['filename']['bad_data'] = inf_config_dict.get('bad_data_filename', change_file_extension(edf_filename, '.evt'))
    encoding_filename = output_path / (edf_file.stem + '.h5')
    hyp['filename']['hypnodensity_pkl'] = change_file_extension(encoding_filename, '.hypnodensity.pkl')
    hyp['filename']['hypnodensity_h5'] = change_file_extension(encoding_filename, '.hypnodensity.h5')
    hyp['filename']['hypnodensity_txt'] = change_file_extension(encoding_filename, '.hypnodensity.txt')
    hyp['filename']['hypnogram'] = change_file_extension(encoding_filename, '.hypnogram.txt')
    hyp['filename']['hypnogram_30_sec'] = change_file_extension(encoding_filename, '.hypnogram.sta')
    hyp['filename']['features_h5'] = change_file_extension(encoding_filename, '.features.h5')
    hyp['filename']['features_pkl'] = change_file_extension(encoding_filename, '.features.pkl')
    hyp['filename']['diagnosis'] = change_file_extension(encoding_filename, '.diagnosis.txt')
    hyp['filename']['plot'] = change_file_extension(encoding_filename, '.hypnodensity.png')
    hyp['filename']['encoding'] = encoding_filename
    hyp['filename']['pkl_encoding'] = None
    hyp['filename']['h5_encoding'] = None
    hyp['filename']['audit'] = None
    # hyp['filename']['encoding'] = change_file_extension(edf_filename, '.h5')

    for key in hyp.keys():
        hyp[key].update(config_input.get(key, {}))

    # The ['save']['hypnodensity'] will overwrite all other hypnodensity_* save options if it has been set
    if hyp['save']['hypnodensity'] is not None:
        hyp['save']['hypnodensity_txt'] = hyp['save']['hypnodensity']
        hyp['save']['hypnodensity_h5'] = hyp['save']['hypnodensity']
        hyp['save']['hypnodensity_pkl'] = hyp['save']['hypnodensity']

    if hyp['save']['features'] is not None:
        hyp['save']['features_h5'] = hyp['save']['features']
        hyp['save']['features_pkl'] = hyp['save']['features']

    # hyp['save'].update(config_input.get('save', {}))
    # hyp['show'].update(config_input.get('show', {}))
    # hyp['filename'].update(config_input.get('filename', {}))

    hypno_config = hyp

    # Auditing filename is used to identify if audit should be done (None: False) and if so, which file to send audit
    # data to (filename: True)
    app_config.filename = hyp['filename']
    # app_config.auditFilename = hyp['filename']['audit']
    # app_config.h5_encoding = hyp['filename']['h5_encoding']
    # app_config.pkl_encoding = hyp['filename']['pkl_encoding']

    app_config.saveEncoding = hyp['save']['encoding']
    app_config.save_hypnodensity_txt = hyp['save']['hypnodensity_txt']
    app_config.save_hypnodensity_h5 = hyp['save']['hypnodensity_h5']
    app_config.save_hypnodensity_pkl = hyp['save']['hypnodensity_pkl']
    app_config.saveHypnodensity = app_config.save_hypnodensity_h5 or app_config.save_hypnodensity_txt or app_config.save_hypnodensity_pkl

    app_config.encodeFilename = hyp['filename']['encoding']
    app_config.encodeOnly = not (hyp['show']['hypnogram'] or hyp['save']['hypnogram']
                                 or hyp['show']['hypnogram_30_sec'] or hyp['save']['hypnogram_30_sec']
                                 or hyp['show']['hypnodensity'] or app_config.saveHypnodensity
                                 or hyp['show']['diagnosis'] or hyp['save']['diagnosis']
                                 or hyp['show']['plot'] or hyp['save']['plot'])
    app_config.hypnodensitySaveOnly = app_config.saveHypnodensity and \
                                      not(hyp['show']['hypnogram'] or hyp['save']['hypnogram']
                                          or hyp['show']['hypnogram_30_sec'] or hyp['save']['hypnogram_30_sec']
                                          or hyp['show']['hypnodensity']
                                          or hyp['show']['diagnosis'] or hyp['save']['diagnosis']
                                          or hyp['show']['plot'] or hyp['save']['plot'])

    app_config.diagnosisSaveOnly = hyp['save']['diagnosis'] and not (hyp['show']['hypnogram'] or hyp['show']['hypnogram_30_sec']
                                 or hyp['show']['hypnodensity'] or hyp['show']['diagnosis']
                                 or hyp['show']['plot'] or hyp['save']['plot'])

    prediction = None
    diagnosis = None
    if app_config.encodeOnly and Path(app_config.encodeFilename).exists():
        logger.debug('Skipping.  Encoding file already exists: %s', app_config.encodeFilename)
    elif app_config.hypnodensitySaveOnly and Path(hyp['filename']['hypnodensity_h5']).exists():
        logger.debug('Skipping.  Hypnodensity file already exists: %s', str(hyp['filename']['hypnodensity_pkl']))
    elif app_config.diagnosisSaveOnly and Path(hyp['filename']['diagnosis']).exists():
        logger.debug('Skipping.  Diagnosis output file file already exists: %s', str(hyp['filename']['diagnosis']))
    else:
        narco_app = NarcoApp(app_config)
        # narco_app.eval_all()
        narco_app.eval_hypnodensity()

        narco_app.get_hypnodensity()

        if narco_app.config.audit.get('diagnosis', False):
            narco_app.audit(narco_app.eval_narcolepsy, 'Diagnosing...')

        if hypno_config['show']['hypnogram']:
            print("Hypnogram:")
            hypnogram = narco_app.get_hypnogram()
            np.set_printoptions(threshold=10000, linewidth=150)  # use linewidth = 2 to output as a single column
            print(hypnogram)

        if hypno_config['show']['hypnogram_30_sec']:
            print("Hypnogram (30 second epochs):")
            hypnogram = narco_app.get_hypnogram(epoch_len=30)
            np.set_printoptions(threshold=10000, linewidth=150)  # use linewidth = 2 to output as a single column
            print(hypnogram)

        # This is the text portion
        if hypno_config['save']['hypnogram']:
            narco_app.save_hypnogram(filename=hypno_config['filename']['hypnogram'])

        if hypno_config['save']['hypnogram_30_sec']:
            narco_app.save_hypnogram(filename=hypno_config['filename']['hypnogram_30_sec'], epoch_len=30)

        if hypno_config['show']['hypnodensity']:
            print("Hypnodensity:")
            hypnodensity = narco_app.get_hypnodensity()
            np.set_printoptions(threshold=10000*5, linewidth=150)
            print(hypnodensity)

        if hypno_config['save']['hypnodensity_txt']:
            narco_app.save_hypnodensity(filename=hypno_config['filename']['hypnodensity_txt'])

        # The following portion concerns the narcolepsy classification partion
        requires_features = hypno_config['save']['features_h5'] or hypno_config['save']['features_pkl'] or hypno_config['show']['diagnosis'] or hypno_config['save']['diagnosis']
        if requires_features:

            narco_app.calculate_all_hypnodensity_features(import_ok=True)
            if hypno_config['save']['features_h5'] or hypno_config['save']['features_pkl']:
                if hypno_config['save']['features_h5']:
                    narco_app.save_features(filename=hypno_config['filename']['features_h5'])
                if hypno_config['save']['features_pkl']:
                    narco_app.save_features(filename=hypno_config['filename']['features_pkl'])

            if hypno_config['show']['diagnosis']:
                print(narco_app.get_diagnosis())

            if hypno_config['save']['diagnosis']:
                narco_app.save_diagnosis(filename=hypno_config['filename']['diagnosis'])

        # This concerns what is displayed to the screen
        if not app_config.encodeOnly:
            render_hypnodensity(narco_app.get_hypnodensity(), show_plot=hypno_config['show']['plot'],
                                save_plot=hypno_config['save']['plot'], filename=hypno_config['filename']['plot'])
        if hyp['show']['diagnosis'] or hyp['save']['diagnosis']:
            prediction = float(narco_app.narcolepsy_probability[0])
            diagnosis = DIAGNOSIS[int(prediction >= NARCOLEPSY_PREDICTION_CUTOFF)]
            logger.debug('Score:  %0.4f.  Diagnosis: %s', prediction, diagnosis)
        else:
            prediction = None
            diagnosis = None

    return prediction, diagnosis


def time2elapsedseconds(edf_file, time_value):
    if isinstance(time_value, str) and ":" in time_value:
        if edf_file is None or not edf_file.exists():
            raise(ValueError('Cannot convert time stamp to elapsed seconds from the study start because an EDF file, which contains the study start time, was not found.'))
        else:
            study_start_time_seconds = edf_file.start_time
            elapsed_seconds = study_start_time_seconds
            return elapsed_seconds
    else:
        return time_value

def change_file_extension(fullname, new_extension):
    basename, _ = os.path.splitext(fullname)
    return basename + new_extension


def render_hypnodensity(hypnodensity, show_plot=False, save_plot=False, filename='tmp.png'):

    if show_plot or save_plot:
        # Remove any rows with nan values
        hypnodensity = hypnodensity[~np.isnan(hypnodensity[:, 0]), :]
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

    edf_filename: Path
    _hypnodensity: Hypnodensity

    def __init__(self, app_config):

        # appConfig is an instance of AppConfig class, defined in inf_config.py
        self.config = app_config
        self.edf_filename = app_config.edf_filename  # full filename of an .EDF to use for header information.  A template .edf

        self._hypnodensity = Hypnodensity(app_config)
        self.narco_model = NarcoModel(app_config, self._hypnodensity)
        self.models_used = app_config.models_used
        self.selected_features = app_config.narco_prediction_selected_features
        self.narcolepsy_probability = []
        self.num_induction_points = 350

    def get_diagnosis(self):
        prediction = self.narcolepsy_probability
        if not prediction:
            prediction = self.get_narco_prediction()
        return "Score: %0.4f\nDiagnosis: %s" % \
               (prediction[0], DIAGNOSIS[int(prediction >= NARCOLEPSY_PREDICTION_CUTOFF)])

    def get_hypnodensity(self):
        return self._hypnodensity.get_hypnodensity()

    def get_hypnogram(self, epoch_len: int = 15):
        return self._hypnodensity.get_hypnogram(epoch_len)

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

    def save_hypnogram(self, filename='', epoch_len: int = 15):
        if filename == '':
            if epoch_len == 30:
                # for 30 second epochs
                filename = change_file_extension(self.edf_path, '.hypnogram.sta')
            else:
                filename = change_file_extension(self.edf_path, '.hypnogram.txt')

        hypno = self.get_hypnogram(epoch_len)
        np.savetxt(filename, hypno, delimiter=",", fmt='%i')

    def save_features(self, filename=''):
        if filename == '':
            filename = change_file_extension(self.edf_path, 'features.h5')
        self._hypnodensity.export_features(Path(filename))

    def audit(self, method_to_audit, audit_label, *args):
        start_time = time.time()
        method_to_audit(*args)
        elapsed_time = time.time() - start_time
        with Path(self.config.filename['audit']).open('a') as fp:
            audit_str = f', {audit_label}: {elapsed_time:0.3f} s'
            fp.write(audit_str)

    def get_narco_models(self):
        return self.narco_model.get_model_names()

    def get_num_hypnodensity_models_used(self):
        return self._hypnodensity.get_num_hypnodensities()

    def get_hypnodensity_model_names(self):
        return [self._hypnodensity._hypnodensity_features.keys()]

    def calculate_all_hypnodensity_features(self, import_ok: bool = True):
        _features = None
        if import_ok:
            h = Path(self.config.filename["features_h5"])
            if h.exists():
                _features = self._hypnodensity.import_model_features(h)
            else:
                h = Path(self.config.filename["features_pkl"])
                if h.exists():
                    _features = self._hypnodensity.import_model_features(h)
        if _features is None:
            _features = dict()
            print(f'Calculating features from hypnodensity')

        for idx, gp_model in enumerate(self.get_narco_models()):
            if gp_model not in _features:
                logger.info(f'Calculating features from hypnodensity[{gp_model}]')
                self.get_all_hypnodensity_features(gp_model, idx)

    # These are all of the features derived from the hypnodensity.
    # args include model_name: str, model_idx: int  (the model index)
    def get_all_hypnodensity_features(self, *args):
        return self.narco_model.get_features(*args)

    # These are the curated set of hypnodensity features used for modeling narcolepsy
    # args include model_name: str, model_idx: int  (the model index)
    def get_narcolepsy_features(self, *args):
        return self.narco_model.get_selected_features(*args)

    def get_narco_prediction(self):  # ,current_subset, num_subjects, num_models, num_folds):
        self.narcolepsy_probability = self.narco_model.get_prediction()
        print(self.narcolepsy_probability[0])
        return self.narcolepsy_probability

    def eval_hypnodensity(self):
        self._hypnodensity.evaluate()

    def eval_narcolepsy(self):
        self.get_narco_prediction()

    def eval_all(self):
        self.eval_hypnodensity()
        self.eval_narcolepsy()


if __name__ == '__main__':
    outputFormat = 'json'

    if sys.argv[1:]:  # if there are at least three arguments (two beyond [0])

        _edf_filename = sys.argv[1]

        # For hard coding/bypassing json input argument, uncomment the following: jsonObj = json.loads('{
        # "channel_indices":{"centrals":[3,4],"occipitals":[5,6],"eog_l":7,"eog_r":8,"chin_emg":9},
        # "show":{"plot":false,"hypnodensity":false,"hypnogram":false}, "save":{"plot":false,"hypnodensity":true,
        # "hypnogram":true}}')
        json_str = json.loads(sys.argv[2])
        try:
            main(_edf_filename, json_str)
        except OSError as oserr:
            print("OSError:", oserr)
    else:
        print(sys.argv[0], 'requires two arguments when run as a script')

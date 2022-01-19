import os
from typing import List, Any

import numpy as np
from pathlib import Path
from inf_narco_features import HypnodensityFeatures


class AppConfig(object):
    edf_file: Path

    def __init__(self):
        # Model folder
        self.models_used = ['ac_rh_ls_lstm_01', 'ac_rh_ls_lstm_02',
                            'ac_rh_ls_lstm_03', 'ac_rh_ls_lstm_04',
                            'ac_rh_ls_lstm_05', 'ac_rh_ls_lstm_06',
                            'ac_rh_ls_lstm_07', 'ac_rh_ls_lstm_08',
                            'ac_rh_ls_lstm_09', 'ac_rh_ls_lstm_10',
                            'ac_rh_ls_lstm_11', 'ac_rh_ls_lstm_12',
                            'ac_rh_ls_lstm_13', 'ac_rh_ls_lstm_14',
                            'ac_rh_ls_lstm_15', 'ac_rh_ls_lstm_16']

        # Uncomment the following when running validation comparison given in readme file.
        # self.models_used = ['ac_rh_ls_lstm_01']

        # Hypnodensity classification settings
        self.relevance_threshold = 1

        # Predictions above this value will be considered narcoleptic
        self.narco_threshold = 0.0

        self.fs = np.array(100, dtype=float)
        self.fs_high = np.array(0.2, dtype=float)
        self.fs_low = np.array(49, dtype=float)

        # Wrapper hooks for auditing processing time of various parts of the app.
        self.audit = {'encoding': False, 'hypnodensity': False, 'diagnosis': False}
        self.channels = ['C3', 'C4', 'O1', 'O2', 'EOG-L', 'EOG-R', 'EMG', 'A1', 'A2']

        # Size of cross correlation in seconds - so in samples this will be
        # sum([200 200 400 400 40 ]) == 1240 + 400 (for EOGLR) == 1640
        self.cc_size = {'C3': 2, 'C4': 2, 'O1': 2, 'O2': 2,
                        'EOG-L': 4, 'EOG-R': 4,
                        'EMG': 0.4,
                        'A1': [], 'A2': [],
                        }

        self.channels_used = dict.fromkeys(self.channels)
        self.loaded_channels = dict.fromkeys(self.channels)

        # Make it easier to run stanford-stages code from other repositories, or scripts in other directories.
        this_path = Path(__file__).parent.absolute()
        self.psg_noise_file_pathname = str(this_path.joinpath('ml/noiseM.mat'))
        self.hypnodensity_model_root_path = str(this_path.joinpath('ml/ac'))
        self.hypnodensity_scale_path = str(this_path.joinpath('ml/scaling/'))

        # self.psg_noise_file_pathname = 'F:/ml/noiseM.mat'
        # self.hypnodensity_model_root_path = 'F:/ml/'
        # self.hypnodensity_scale_path = 'F:/ml/scaling/'

        # Related to classifying narcolepsy from hypnodensity features
        self.narco_classifier_path: str = str(this_path.joinpath('ml/gp/'))

        self.edf_file = None

        # lights off and on indicate the number of seconds from the start of the psg recording until the
        # "lights off" and "lights on" events for a formal psg recording.  Negative values indicate time
        # elapsed from the end of the study.  For example: lights_on=-1 means the lights were turned on 1 s
        # prior to the recording ending.  Leave values as None to include all data from the start of the
        # recording until the end of the reocrding, otherwise only the time between lights_off until
        # lights_on will be evaluated for hypnodensity features to be used in narcolepsy detection, and
        # for sleep staging scoring.  Epochs outside lights off/on range and lights on epoch itself are
        # scored as '7' in the hypnogram to indicate unscored epochs.  Similiarly, these epochs are
        # excluded from the features extraction step that is used to predict narcolepsy.
        self._lights_off: int = None
        self._lights_on: int = None

        # Set to 0 to turn off cross-validation.  Otherwise a model must have been trained previously for each fold desired.  This is for the gp classifier.
        self.narco_prediction_num_folds = 0

        self.narco_prediction_scales = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self.narco_prediction_selected_features = None
        # Uncomment to use the following 153 features
        # self.narco_prediction_selected_features = [0, 2, 4, 8, 9, 10, 13, 15, 16, 17, 19, 21, 24, 25, 28, 31, 32, 34, 35, 39, 46, 47, 50, 52, 58, 61, 63, 64, 77, 93, 102, 110, 119, 121, 122, 123, 126, 127, 131, 133, 137, 150, 154, 156, 157, 160, 162, 166, 167, 173, 176, 181, 182, 192, 195, 197, 199, 205, 206, 208, 210, 215, 220, 223, 227, 236, 237, 245, 246, 257, 258, 260, 262, 267, 273, 274, 275, 281, 284, 285, 288, 293, 294, 295, 302, 311, 316, 317, 319, 321, 331, 333, 334, 337, 341, 348, 350, 356, 357, 362, 363, 364, 368, 369, 371, 373, 376, 378, 380, 381, 387, 390, 392, 393, 394, 398, 400, 403, 404, 406, 407, 412, 414, 415, 419, 422, 428, 429, 436, 437, 441, 442, 443, 445, 447, 454, 455, 456, 457, 459, 460, 461, 462, 463, 469, 470, 471, 474, 475, 477, 481, 482, 485]

        # Feature scaling options include:
        # ['range']:           (x-median(X))/(percentile(X, 85)-percentile(X, 15)) where x is the feature and X is the population sample of the features
        # 'z':                 (x-mean(X))/std(X)  - This is gives 0 mean and unit variance.
        # 'unscaled' or None:  (x - 0) / 1 - This is unscaled; no change to the features.
        self.narco_feature_scaling_method = 'range'

        # Set to False to minimize printed output.
        self.verbose: bool = True

    @staticmethod
    def str2int(value):
        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                value = None
        return value

    def set_narco_feature_selection(self, feature_selections):
        if feature_selections is None:
            self.narco_prediction_selected_features = None
        elif isinstance(feature_selections, list):
            self.narco_prediction_selected_features = feature_selections
        elif feature_selections.lower() == 'all':
            self.narco_prediction_selected_features = list(range(HypnodensityFeatures.num_features))
        else:
            print('Unhandled value for feature_selections, which should be a list or "all"')
            exit(1)

    @property
    def lights_off(self):
        return self._lights_off

    @lights_off.setter
    def lights_off(self, value: int):
        self._lights_off = self.str2int(value)
        if self._lights_off is None:
            self._lights_off = 0

    @property
    def lights_on(self):
        return self._lights_on

    @lights_on.setter
    def lights_on(self, value: int):
        self._lights_on = self.str2int(value)
        if self.lights_on == 0:
            self.lights_on = None

    def get_lights_off_epoch(self, epoch_len: int = 15):
        return self.sec2epoch(self.lights_off, epoch_len)

    def get_lights_on_epoch(self, epoch_len: int= 15):
        return self.sec2epoch(self.lights_on, epoch_len)

    @staticmethod
    def sec2epoch(sec: int, epoch_len: int = 15):
        # Translates an integer value in seconds to the equivalent epoch based on the radix epoch_len,
        # which is also given in seconds.  This works here for negative values as well, which are assumed to be
        # referenced from the end of the study, where a -1 sec would be the equivalent of -1 epoch,
        # which includeds the last epoch of the study using python indexing (e.g. study[-1]).
        # Returns None if sec is not entered or None
        if sec is None:
            return None
        else:
            return sec // epoch_len # np.floor_divide(sec, epoch_len)


# Define Config
class Config(object):

    @staticmethod
    def get(model_name):
        if model_name[0:2] == 'ac':
            return ACConfig(model_name)
        else:
            raise Exception

    def __getitem__(self, itemname):
        return object.__getattribute__(self, itemname)

    def __init__(self, scope, num_features, num_hidden, segsize, lstm, num_classes, batch_size, max_train_len, atonce,
                 restart=True, model_name='small_lstm', is_train=False, root_model_dir='./'):

        self.hypnodensity_model_dir = os.path.join(root_model_dir, scope, model_name)
        self.model_name = model_name
        self.scope = scope

        self.is_training = is_train
        self.num_features = num_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.restart = restart
        self.lstm = lstm
        self.num_hidden = num_hidden
        self.keep_prob = 0.5
        self.segsize = segsize
        self.eval_nseg_atonce = atonce
        self.max_train_len = max_train_len
        self.save_freq = 2
        self.max_steps = 2000
        self.init_learning_rate = 0.005
        self.learning_rate_decay = 750

    def checkpoint_file(self, ckpt=0):
        if ckpt == 0:
            return os.path.join(self.hypnodensity_model_dir, 'model.ckpt')
        else:
            return os.path.join(self.hypnodensity_model_dir, 'model.ckpt-%.0f' % ckpt)


# Now we can pass Config to ACConfig for construction; inheritence.
class ACConfig(Config):

    def __init__(self, restart=True, model_name='ac_rh_ls_lstm_01', is_training=False, root_model_dir='./'):

        print('model: ' + model_name)
        if model_name[3:5] == 'lh':
            num_hidden = 256
        elif model_name[3:5] == 'rh':
            np.random.seed(int(model_name[-2:]))
            num_hidden = 256 + np.round(np.random.rand(1) * 128)
            num_hidden = num_hidden[0].astype(int)
        else:
            num_hidden = 128

        if model_name[6:8] == 'ls':
            segsize = 60
            atonce = 1000  # 2 batches, runs out of memory though :(
            # atonce = 1500  #
            # atonce = 600 # 173 seconds
            # atonce = 500  # 154 seconds
            # atonce = 400 # 154 seconds
            # atonce = 200 # 150 seconds
            # atonce = 100 # 110, 121
            # atonce = 60 # 120 (num batches = 30)
            # atonce = 50 # 104
            # atonce = 45  # 102, 116 (allow growth = true, num batches = 40
            # atonce = 40 # 101, 123, 114 (allow growth=true) numbatches = 44
            # atonce = 35 # 103, 132
            # atonce = 30 # 102, 116, allow growth=true, num batches = 59
            # atonce = 10 # 118, 133 (num batches = 176)
        else:
            segsize = 20
            atonce = 3000

        if model_name[9:11] == 'ff':
            lstm = False
        else:
            lstm = True

        if is_training:
            batch_size = 30
        else:
            batch_size = 1

        scope = 'ac'
        num_features = 1640
        num_classes = 5
        max_train_len = 14400
        super(ACConfig, self).__init__(scope, num_features, num_hidden, segsize, lstm, num_classes, batch_size,
                                       max_train_len, atonce, restart, model_name, is_training, root_model_dir)

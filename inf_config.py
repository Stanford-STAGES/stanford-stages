import os
import numpy as np


class AppConfig(object):

    def __init__(self):

        # Model folder
        self.models_used = ['ac_rh_ls_lstm_01']

        # Uncomment the following when running validation comparison given in readme file.
        # self.models_used = ['ac_rh_ls_lstm_01']

        # Hypnodensity classification settings
        self.relevance_threshold = 1
        self.fs = np.array(100,dtype=float)
        self.fsH = np.array(0.2,dtype=float)
        self.fsL = np.array(49,dtype=float)

        self.channels = ['C3','C4','O1','O2','EOG-L','EOG-R','EMG','A1','A2']

        # Size of cross correlation in seconds - so in samples this will be sum([200 200 400 400 40 ]) == 1240 + 400 for EOGLR == 1640
        self.CCsize = {'C3':   2, 'C4':   2,  'O1':   2, 'O2':   2,
                       'EOG-L':4, 'EOG-R':4,
                       'EMG':  0.4,
                       'A1':   [], 'A2':   [],
                       }
        #self.CCsize = dict(zip(self.channels,
        #                [2,2,2,2,4,4,0.4,[],[]]))
        self.channels_used = dict.fromkeys(self.channels)
        self.loaded_channels = dict.fromkeys(self.channels)

        self.psg_noise_file_pathname = './ml/noiseM.mat'
        self.hypnodensity_model_root_path = './ml/'
        self.hypnodensity_scale_path = './ml/scaling/'
        # self.hypnodensity_select_features_path = './ml/'
        # self.hypnodensity_select_features_pickle_name = 'narcoFeatureSelect.p'

        self.Kfold = 10  # or 20
        self.edf_path = []
        self.lightsOff = []
        self.lightsOn = []

        # Related to classifying narcolepsy from hypnodensity features
        self.narco_classifier_path = './ml/gp'

        self.narco_prediction_num_folds = 5 # for the gp narco classifier
        self.narco_prediction_scales = [0.90403101, 0.89939177, 0.90552177, 0.88393560,
          0.89625522, 0.88085868, 0.89474061, 0.87774597,
          0.87615981, 0.88391175, 0.89158020, 0.88084675,
          0.89320215, 0.87923673, 0.87615981, 0.88850328]

        self.narco_prediction_selected_features = [1, 11, 16, 22, 25, 41, 43, 49, 64, 65, 86, 87, 103, 119, 140,
                                                   147, 149, 166, 196, 201, 202, 220, 244, 245, 261, 276, 289, 296,
                                                   299, 390, 405, 450, 467, 468, 470, 474, 476, 477]


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
                 restart=True, model_name='small_lstm', is_train=False,
                 root_model_dir = './',  # Change this if models are saved elsewhere
                ):

        self.hypnodensity_model_dir = os.path.join(root_model_dir, scope, model_name)

        # Data

        # Configuration
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

    def __init__(self, restart=True, model_name='ac_rh_ls_lstm_01', is_training=False, root_model_dir = './'):

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
            atonce = 1000
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

import numpy as np
import tensorflow as tf
import gpflow
import os
from datetime import datetime
from pathlib import Path
from inf_config import AppConfig
from inf_hypnodensity import Hypnodensity
import logging
import warnings

warnings.simplefilter('ignore')
gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ModelLoadException(Exception):
    pass


class ModelCheckPointError(Exception):
    pass


class NarcoModel(object):
    """
      This <strike>trains a SVGP for classifying narcolepsy or</strike> loads a previously trained SVGP for classifying narcolepsy.
      The algorithm produces values between −1 and 1, with 1 indicating a high probability of narcolepsy.
      The cut-off threshold between narcolepsy type 1 and “other“ is set at 0.0 currently.  The manuscript sets it
      at −0.03 (-0.03, 1.0] being narcolepsy
      """

    # Added to stanford-stages: 7/14/2021
    # Author: Hyatt Moore, IV
    _num_folds: int
    _models_parent_path: str
    _model_path: str

    def __init__(self, app_config: AppConfig = None, hypnodensity_obj: Hypnodensity = None):
        if app_config is None:
            app_config = AppConfig()

        self._m: gpflow.models.SVGP = None
        self._hypnodensity_obj: Hypnodensity = hypnodensity_obj
        self.edf_filename = 'ref_edf_filename_placeholder'

        # For check pointing method - which is being phased out
        self._kernel = None
        self._likelihood = gpflow.likelihoods.Gaussian()

        try:
            self._selected_features = app_config.narco_prediction_selected_features
        except:
            self._selected_features = []

        self.num_folds = app_config.narco_prediction_num_folds = 0
        self.use_crossvalidation = False

        self.set_models_parent_path(app_config.narco_classifier_path)

        self.config = app_config

    # @staticmethod
    # def defaults(output_json: bool = False) -> {}:
    #     _defaults = {
    #         'GP_MODELS_PATH': 'C:/Data/OAK/narco_gp_models',
    #         'NUM_FOLDS': 5,
    #         'NOISE_REFERENCE_FILE': "F:\\ml\\noiseM.mat",
    #         'SCALING_PATH': 'X:/ml/scaling'  # for scaling hypnodensity features used in narcolepsy classification
    #     }
    #     if output_json:
    #         _defaults = json.dumps(_defaults, sort_keys=True, indent=4)
    #     return _defaults

    @property
    def num_folds(self) -> int:
        return self._num_folds

    @num_folds.setter
    def num_folds(self, num: int):
        if isinstance(num, int):
            if num >= 0:
                self._num_folds = num
                if num == 0:
                    self.use_crossvalidation = False
            else:
                raise ValueError
        else:
            raise TypeError

    # Individual models are saved as subpaths of the parent path
    def set_models_parent_path(self, models_parent_path):
        if not isinstance(models_parent_path, Path):
            models_parent_path = Path(models_parent_path)
        self._models_parent_path = models_parent_path

    def set_feature_selection(self, feature_selection):
        self._selected_features = feature_selection

    # Typically the list of model names specified in the configuration fille (e.g. ['ac_rh_ls_lstm_01','ac_rh_ls_lstm_02', ...]
    def get_model_names(self):
        return self.config.models_used

    def load_model(self, filename):
        if isinstance(filename, Path):
            filename = str(filename)
        self._m = tf.saved_model.load(filename)

    # checkpointing version
    def load_model_checkpoint(self, load_path=None):
        if load_path is not None:
            # with tf.Graph().as_default() as graph:
            #     with tf.Session():
            #         m = gpflow.saver.Saver().load(os.path.join(gpmodels_base_path, gpmodel, gpmodel + '_fold{:02}.gpm'.format(k+1)))
            #         mean_pred[:, idx, k, np.newaxis], var_pred[:, idx, k, np.newaxis] = m.predict_y(X),

            # self._m = gpflow.models.SVGP(kernel_self._kernel, )
            self._kernel = gpflow.kernels.RationalQuadratic(len(self._selected_features), ard=True)
            z = np.ones(shape=(self._num_induction_points, len(self._selected_features)), dtype=np.float64)
            self._m = gpflow.models.SVGP(kernel=self._kernel, likelihood=self._likelihood, inducing_variable=z)
            ckpt: tf.train.Checkpoint = tf.train.Checkpoint(model=self._m)
            # ckpt.restore(load_path)
            manager: tf.train.CheckpointManager = tf.train.CheckpointManager(ckpt, directory=load_path, max_to_keep=5)
            if manager.latest_checkpoint is None:
                raise ModelCheckPointError()
            else:
                try:
                    status = ckpt.restore(manager.latest_checkpoint).expect_partial()
                except ValueError as err:
                    self._num_induction_points = err
                    z = np.ones(shape=(self._num_induction_points, len(self._selected_features)), dtype=np.float64)
                    self._m = gpflow.models.SVGP(kernel=self._kernel, likelihood=self._likelihood, inducing_variable=z)
                    ckpt: tf.train.Checkpoint = tf.train.Checkpoint(model=self._m)
                    # ckpt.restore(load_path)
                    manager: tf.train.CheckpointManager = tf.train.CheckpointManager(ckpt, directory=load_path,
                                                                                     max_to_keep=5)
                    status = ckpt.restore(manager.latest_checkpoint).expect_partial()
                # print(status)

    def get_selected_features(self, model_name: str, idx: int):
        """
        :param model_name: String ID of the model.  This identifies the scale factor to apply to the features.
        :param idx: The numeric index of the model being used.  This identifies the hypnodensity to gather features from
        :return: The selected, extracted, and scaled features for hypnodensity derived using the specified model index
        (idx) between [lights_off, lights_on).  Note: [inclusive, exclusive).  The end.
        """
        print('Getting selected features')
        x = self.get_features(model_name, idx, scale_features=True)
        selected_features = self.config.narco_prediction_selected_features
        print('Size of x is', x.shape,'Size of selected_features is', len(selected_features))
        return x[:, selected_features]

    def get_features(self, model_name: str, idx: int, scale_features: bool = False):
        '''
        Extracts all narcolepsy model features using hypnodensity as input.  The narcolepy features are stored in the
        hypnodenisty_features property, a dictionary keyed on the model_name.
        :param model_name:
        :param idx:
        :param scale_features: Set to true to scale features using the scale values found in the pickle file setup in the configuration property.
        Default is False so that scalar values can be determined more readily in new datasets.
        :return:
        '''
        # check if we already have them or if they can be loaded...

        # if we already have them
        if model_name in self._hypnodensity_obj._hypnodensity_features:
            x = self._hypnodensity_obj._hypnodensity_features[model_name]
        else:
            x = self._hypnodensity_obj.import_model_features(model_name)

        if x is None:
            _hypnodensity = self._hypnodensity_obj.hypnodensity[idx]
            epoch_len: int = 15
            bad_signal_events_1_sec = self._hypnodensity_obj.get_signal_quality_events()
            bad_signal_epoch_len = [self._hypnodensity_obj.epoch_rebase(x, 1, epoch_len).astype(np.uint32) for x in
                                    bad_signal_events_1_sec]
            # Consider also -->  bad_signal_epoch_len = self.config.sec2epoch(bad_signal_events_1_sec, epoch_len)

            # remove any sections identified with flatline or other bad signal data found
            for start_stop in bad_signal_epoch_len:
                _hypnodensity[start_stop[0]:start_stop[1] + 1, :] = np.nan

            lights_off_epoch = self.config.get_lights_off_epoch(epoch_len=epoch_len)
            lights_on_epoch = self.config.get_lights_on_epoch(epoch_len=epoch_len)

            # Only consider data between lights off and lights on.
            _hypnodensity = _hypnodensity[lights_off_epoch:lights_on_epoch, :]

            # self.hypnodensity is a list of numpy arrays.
            # _hypnodensity = self.hypnodensity[idx][lights_off_epoch:lights_on_epoch, :]
            # configuration is currently setup for 15 second epochs (magic).
            # segments are .25 second and we have 60 of them
            x = self._hypnodensity_obj.features.extract(_hypnodensity)
            self._hypnodensity_obj._hypnodensity_features[model_name] = x

        if scale_features:
            x = self._hypnodensity_obj.features.scale_features(x, model_name)
        return x

    def predict_y(self, x: np.ndarray, threshold: float = 0):

        mean_pred, var_pred = self._m.predict_y_compiled(tf.convert_to_tensor(x, dtype=tf.float64))
        # mean_pred, var_pred = self._m.predict_y(x)

        # Get the prediction accuracy at 50% cutoff
        y_pred = np.ones(mean_pred.shape)
        y_pred[mean_pred < threshold] = -1
        return y_pred, mean_pred, var_pred

    def get_accurracy(self, x_test: np.ndarray, y_test: np.ndarray):
        # y_pred, _ = self.predict_y(x_test)
        y_pred = self.predict_y(x_test)[0]
        acc = np.mean(np.squeeze(y_pred) == np.squeeze(y_test))
        return acc

    # Use get_model_features with scale flag set to true/false depending if you want features scaled or not.  default is False
    def get_prediction(self):
        scales = self.config.narco_prediction_scales
        narco_pred = np.nan

        # Used to do checkpointing for file load and save.  Now using tf.saved_model
        uses_checkpointing = False
        uses_crossvalidation = self.use_crossvalidation

        gp_models_base_path = self.config.narco_classifier_path
        gp_models_requested = self.get_model_names()
        num_folds = self.config.narco_prediction_num_folds
        num_models_requested = len(gp_models_requested)

        if num_models_requested == 0:
            raise ModelLoadException('No narcolepsy models requested in configuration file.  Prediction is not possible', self.edf_filename)

        num_paths_expected = num_models_requested * num_folds

        #gp_models = {gp_model: os.path.join(gp_models_base_path, gp_model) for gp_model in gp_models_requested if
        #             os.path.exists(os.path.join(gp_models_base_path, gp_model))}

        # Pre-checking here to see if we have all of the models paths necessary before moving on to processing them.
        if uses_crossvalidation:
            gp_models = []
            for gp_model in gp_models_requested:
                for k in range(0, num_folds):
                    if uses_checkpointing:
                        cv_models = [k for k in range(0, num_folds) if
                                     os.path.exists(os.path.join(gp_models_base_path, gp_model, str(k)))]
                    else:
                        cv_models = [k for k in range(0, num_folds) if
                                     (gp_models_base_path / (gp_model + '_cv_' + str(k))).is_dir()]
                # Tally models which have all folds expected.
                if len(cv_models) == num_folds:
                    gp_models.append(gp_model)
                else:
                    logger.error('Missing %d models folds for %s', num_folds-cv_models, gp_model)
        else:
            # Overwrite num folds with 1 if we are not using cross-validation
            num_folds = 1
            gp_models = [gp_model for gp_model in gp_models_requested if os.path.exists(os.path.join(gp_models_base_path, gp_model))]

        num_models_found = len(gp_models)

        # Proxy for how many feature sets we will have for our ensemble.
        num_hypnodenisty_models = self._hypnodensity_obj.get_num_hypnodensities()

        if num_models_found == 0:
            logger.error(f'No narcolepsy models found for prediction at "{gp_models_base_path}".  Check config file '
                         f'or path.  Stopping!')
            raise ModelLoadException('No narcolepsy models found for prediction', self.edf_filename)

        # Not okay if we find fewer than requested
        elif num_models_found < num_models_requested:

            if uses_crossvalidation:
                logger.error(
                    'Not all model fold paths were found: %d of %d were missing.  Check config file and subpaths '
                    'of "%s".', num_models_requested-num_models_found, num_paths_expected, gp_models_base_path)
            else:
                logger.error('Expecting %d models, but found %d models.  Check config file and path ("%s")',
                             num_models_requested, num_models_found, gp_models_base_path)
            raise ModelLoadException('Missing narcolepsy prediction models.', self.edf_filename)

        # Again, Not okay if we find fewer than requested
        elif num_hypnodenisty_models < num_models_requested:
            logger.error('Narcolepsy prediction expects hypnodensities from %d models, but %d were found.  Check '
                         'config file and path ("%s")',
                         num_models_requested, num_hypnodenisty_models, gp_models_base_path)
            raise ModelLoadException(f'Narcolepsy prediction model count ({num_models_requested}) mismatch with '
                                      f'hypnodensity model count found ({num_hypnodenisty_models})', self.edf_filename)

        # Past initial path checking.
        num_subjects = 1
        narco_pred = np.zeros(shape=(num_subjects, ), dtype=np.float64)
        narco_pred_var = np.zeros(shape=(num_subjects, ), dtype=np.float64)

        narco_threshold = self.config.narco_threshold

        gp_models_base_path = Path(gp_models_base_path)
        for model_idx, gp_model in enumerate(gp_models):
            print('{} | Predicting using: {}'.format(datetime.now(), gp_model))
            print('Getting narcolepsy features for', gp_model)
            x = self.get_selected_features(gp_model, model_idx)

            if uses_crossvalidation:
                cv_acc = 0
                for k in range(num_folds):
                    if uses_checkpointing:
                        model_path = gp_models_base_path / gp_model / str(k)
                        self.narco_model.load_model_checkpoint(load_path=model_path)
                    else:
                        model_filename = gp_models_base_path / (gp_model + '_cv_' + str(k))
                        self.load(filename=model_filename)
                    y_pred_thresh, y_prob = self.predict_y(x, threshold=narco_threshold)
                    narco_pred += (y_prob * self.config.narco_prediction_scales[model_idx]) / np.sum(
                        self.config.narco_prediction_scales) / self.num_folds
            else:
                if uses_checkpointing:
                    model_path = gp_models_base_path / gp_model / "single"
                    self.load_model_checkpoint(load_path=model_path)
                else:
                    model_filename = gp_models_base_path / gp_model
                    self.load_model(filename=model_filename)
                    # potentially raise an error if the model or path has been deleted somehow since starting.
                    # raise StanfordStagesError(f'MISSING Model: {gp_model_fold_pathname}', self.edf_filename)

                y_pred_thresh, y_prob, y_var = self.predict_y(x, threshold=narco_threshold)

                # y_pred is based on the most votes:
                # y_pred += y_pred_thresh / num_models_requested

                # y_pred is based on the probability of the score:
                # y_pred += y_prob / num_models_requested

                # y_pred is based on the probability of the score scaled by the relative accuracy of the model:
                narco_pred += (y_prob * self.config.narco_prediction_scales[model_idx]) / np.sum(
                    self.config.narco_prediction_scales)
                narco_pred_var += (y_var * self.config.narco_prediction_scales[model_idx]) / np.sum(
                    self.config.narco_prediction_scales)
        return narco_pred

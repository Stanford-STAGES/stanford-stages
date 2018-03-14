import inf_config as sc_config
from inf_train import train, get_variable_names
import tensorflow as tf
import os

def main(argv=None):

    config = sc_config.ACConfig(restart=False, model_name="ac_lh_ls_lstm_01", is_training=True)
    curPath = os.path.dirname(os.path.realpath(__file__))
    config.train_data = os.path.join(curPath,'..','data', 'ac_h5_sleep_stage_training')
    # config.test_data = os.path.join('/media/neergaard/neergaardhd/jens', 'ac_testing_data/ac_data_test3')
    train(config)


if __name__ == '__main__':
    print('Tutorial (1): Training sleep stage scoring model');
    tf.app.run();

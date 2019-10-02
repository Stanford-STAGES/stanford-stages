import re
import tensorflow as tf
import numpy as np


def batch_norm(x, n_out, av_dims, is_training, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.compat.v1.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, av_dims, name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=0.99)

    # phase_train = tf.get_variable('is_training',[],dtype=bool,trainable=False,initializer=tf.constant_initializer(True))
    phase_train = tf.constant(True, dtype=bool, name='is_training')
    if not is_training:
        phase_train = tf.logical_not(phase_train, name='is_not_training')

    # phase_train = tf.Print(phase_train,[phase_train])
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    # meanV = tf.Print(mean,[mean])
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


"""
def _activation_summary(x):

  tensor_name = re.sub('%s summary', '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
"""


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.compat.v1.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=True)
        return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.compat.v1.add_to_collection('losses', weight_decay)

    return var


def conv_block(config, inputs, scope_name, fShape, stride):
    with tf.compat.v1.variable_scope(scope_name) as scope:
        kernel = _variable_with_weight_decay('weights', shape=fShape,
                                             stddev=1e-3, wd=0.000001)
        conv = tf.nn.conv2d(inputs, kernel, [1, 1, stride, 1], padding='SAME')
        biases = _variable_on_cpu('biases', fShape[3], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        bnormed = batch_norm(bias, fShape[3], [0, 1, 2], config.is_training, scope=scope_name)
        conv = tf.nn.relu(bnormed, name=scope.name)
        # _activation_summary(conv)

        return conv


def conv2d_block(config, inputs, scope_name, fShape, stride):
    with tf.compat.v1.variable_scope(scope_name) as scope:
        kernel = _variable_with_weight_decay('weights', shape=fShape,
                                             stddev=1e-3, wd=0.00001)
        conv = tf.nn.conv3d(inputs, kernel, [1, 1, stride[0], stride[1], 1], padding='SAME')
        biases = _variable_on_cpu('biases', fShape[4], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        bnormed = batch_norm(bias, fShape[4], [0, 1, 2, 3], config.is_training, scope=scope_name)
        conv = tf.nn.relu(bnormed, name=scope.name)
        # _activation_summary(conv)

        return conv


def small_autocorr(inputs, config, modality, batch_size):
    if modality == 'eeg':
        nIn = 2
        nOut = [64, 128, 256]
        inputs = tf.reshape(inputs, shape=[batch_size, -1, config.segsize, 2, 200])
        inputs = tf.transpose(inputs, perm=[0, 1, 4, 2, 3])

        strides = [[3, 2], [2, 1]]
    elif modality == 'eog':
        nIn = 3
        nOut = [64, 128, 256]
        inputs = tf.reshape(inputs, shape=[batch_size, -1, config.segsize, 3, 400])
        inputs = tf.transpose(inputs, perm=[0, 1, 4, 2, 3])

        strides = [[4, 2], [2, 1]]
    else:
        nIn = 1
        nOut = [16, 32, 64]
        inputs = tf.reshape(inputs, shape=[batch_size, -1, config.segsize, 1, 40])
        inputs = tf.transpose(inputs, perm=[0, 1, 4, 2, 3])

        strides = [[2, 2], [2, 1]]

    conv1 = conv2d_block(config, inputs, 'conv1' + modality, [1, 7, 7, nIn, nOut[0]], strides[0])

    pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                             padding='SAME', name='pool1' + modality)

    conv3 = conv2d_block(config, pool1, 'conv3' + modality, [1, 5, 5, nOut[0], nOut[1]], strides[1])

    conv4 = conv2d_block(config, conv3, 'conv4' + modality, [1, 3, 3, nOut[1], nOut[1]], [1, 1])

    pool2 = tf.nn.max_pool3d(conv4, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1],
                             padding='SAME', name='pool2' + modality)

    conv5 = conv2d_block(config, pool2, 'conv5' + modality, [1, 3, 3, nOut[1], nOut[2]], [1, 1])

    conv6 = conv2d_block(config, conv5, 'conv6' + modality, [1, 3, 3, nOut[2], nOut[2]], [1, 1])

    meanPool = tf.reduce_mean(conv6, 2, name='mean_pool1' + modality)

    meanPool = tf.reduce_mean(meanPool, 2, name='mean_pool2' + modality)

    return meanPool


def large_autocorr(inputs, config, modality, batch_size):
    if modality == 'eeg':
        nIn = 2
        nOut = [64, 128, 256, 512]
        inputs = tf.reshape(inputs, shape=[batch_size, -1, config.segsize, 2, 200])
        inputs = tf.transpose(inputs, perm=[0, 1, 4, 2, 3])

        strides = [[3, 2], [2, 1]]
    elif modality == 'eog':
        nIn = 3
        nOut = [64, 128, 256, 512]
        inputs = tf.reshape(inputs, shape=[batch_size, -1, config.segsize, 3, 400])
        inputs = tf.transpose(inputs, perm=[0, 1, 4, 2, 3])

        strides = [[4, 2], [2, 1]]
    else:
        nIn = 1
        nOut = [16, 32, 64, 512]
        inputs = tf.reshape(inputs, shape=[batch_size, -1, config.segsize, 1, 40])
        inputs = tf.transpose(inputs, perm=[0, 1, 4, 2, 3])

        strides = [[2, 2], [2, 1]]

    conv1 = conv2d_block(config, inputs, 'conv1' + modality, [1, 7, 7, nIn, nOut[0]], strides[0])

    pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                             padding='SAME', name='pool1' + modality)

    conv3 = conv2d_block(config, pool1, 'conv3' + modality, [1, 5, 5, nOut[0], nOut[1]], strides[1])
    conv4 = conv2d_block(config, conv3, 'conv4' + modality, [1, 3, 3, nOut[1], nOut[1]], [1, 1])

    pool2 = tf.nn.max_pool3d(conv4, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1],
                             padding='SAME', name='pool2' + modality)

    conv5 = conv2d_block(config, pool2, 'conv5' + modality, [1, 3, 3, nOut[1], nOut[2]], [1, 1])
    conv6 = conv2d_block(config, conv5, 'conv6' + modality, [1, 3, 3, nOut[2], nOut[2]], [1, 1])

    pool3 = tf.nn.max_pool3d(conv5, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1],
                             padding='SAME', name='pool3' + modality)

    conv7 = conv2d_block(config, pool3, 'conv7' + modality, [1, 3, 3, nOut[2], nOut[3]], [1, 1])
    conv8 = conv2d_block(config, conv7, 'conv8' + modality, [1, 3, 3, nOut[3], nOut[3]], [1, 1])

    meanPool = tf.reduce_mean(conv8, 2, name='mean_pool1' + modality)

    meanPool = tf.reduce_mean(meanPool, 2, name='mean_pool2' + modality)

    return meanPool


def random_autocorr(inputs, config, modality, batch_size):
    if modality == 'eeg':
        np.random.seed(int(config.model_name[-2:]) + 1)
        nIn = 2
        nOut = [np.random.randint(32, 96),
                np.random.randint(64, 192),
                np.random.randint(128, 384)]
        inputs = tf.reshape(inputs, shape=[batch_size, -1, config.segsize, 2, 200])
        inputs = tf.transpose(inputs, perm=[0, 1, 4, 2, 3])

        strides = [[3, 2], [2, 1]]
    elif modality == 'eog':
        np.random.seed(int(config.model_name[-2:]) + 2)
        nIn = 3
        nOut = [np.random.randint(32, 96),
                np.random.randint(64, 192),
                np.random.randint(128, 384)]
        inputs = tf.reshape(inputs, shape=[batch_size, -1, config.segsize, 3, 400])
        inputs = tf.transpose(inputs, perm=[0, 1, 4, 2, 3])

        strides = [[4, 2], [2, 1]]
    else:
        np.random.seed(int(config.model_name[-2:]) + 3)
        nIn = 1
        nOut = [np.random.randint(8, 24),
                np.random.randint(16, 48),
                np.random.randint(32, 96)]
        inputs = tf.reshape(inputs, shape=[batch_size, -1, config.segsize, 1, 40])
        inputs = tf.transpose(inputs, perm=[0, 1, 4, 2, 3])

        strides = [[2, 2], [2, 1]]

    conv1 = conv2d_block(config, inputs, 'conv1' + modality, [1, 7, 7, nIn, nOut[0]], strides[0])

    pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                             padding='SAME', name='pool1' + modality)

    conv2 = conv2d_block(config, pool1, 'conv2' + modality, [1, 5, 5, nOut[0], nOut[1]], strides[1])
    conv3 = conv2d_block(config, conv2, 'conv3' + modality, [1, 3, 3, nOut[1], nOut[1]], [1, 1])

    pool2 = tf.nn.max_pool3d(conv3, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1],
                             padding='SAME', name='pool2' + modality)

    conv4 = conv2d_block(config, pool2, 'conv4' + modality, [1, 3, 3, nOut[1], nOut[2]], [1, 1])
    conv5 = conv2d_block(config, conv4, 'conv5' + modality, [1, 3, 3, nOut[2], nOut[2]], [1, 1])

    meanPool = tf.reduce_mean(conv5, 2, name='mean_pool1' + modality)

    meanPool = tf.reduce_mean(meanPool, 2, name='mean_pool2' + modality)

    return meanPool


def main(inputs, config, modality, batch_size):
    if config.model_name[0:2] == 'ac':
        if config.model_name[3:5] == 'lh':
            hidden = large_autocorr(inputs, config, modality, batch_size)
        elif config.model_name[3:5] == 'rh':
            hidden = random_autocorr(inputs, config, modality, batch_size)
        else:
            hidden = small_autocorr(inputs, config, modality, batch_size)

    return hidden

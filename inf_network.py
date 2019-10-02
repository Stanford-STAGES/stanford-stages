import inf_convolution as sc_conv

import tensorflow as tf
import pdb


class SCModel(object):

    def __init__(self, ac_config):
        self.is_training = ac_config.is_training
        # Placeholders

        self._features = tf.compat.v1.placeholder(tf.float32, [None, None, ac_config.num_features], name='ModelInput')
        self._targets = tf.compat.v1.placeholder(tf.float32, [None, ac_config.num_classes], name='ModelOutput')
        self._mask = tf.compat.v1.placeholder(tf.float32, [None], name='ModelWeights')

        self._batch_size = tf.compat.v1.placeholder(tf.int32, name='BatchSize')
        self._learning_rate = tf.compat.v1.placeholder(tf.float32, name='LearningRate')

        batch_size_int = tf.reshape(self._batch_size, [])

        if ac_config.lstm:
            self._initial_state = tf.compat.v1.placeholder_with_default(
                tf.zeros([batch_size_int, ac_config.num_hidden * 2], dtype=tf.float32),
                [None, ac_config.num_hidden * 2], name='InitialState')

        # Layer in
        with tf.compat.v1.variable_scope('input_hidden') as scope:
            inputs = self._features
            inputs = tf.reshape(inputs, shape=[batch_size_int, -1, ac_config.segsize, ac_config.num_features])

        hidden_eeg = sc_conv.main(inputs[:, :, :, :400], ac_config, 'eeg', batch_size_int)
        hidden_eog = sc_conv.main(inputs[:, :, :, 400:1600], ac_config, 'eog', batch_size_int)
        hidden_emg = sc_conv.main(inputs[:, :, :, 1600:], ac_config, 'emg', batch_size_int)

        # For debugging
        # pdb.set_trace()
        if tf.__version__ < '1.0':
            hidden_combined = tf.concat(2, [hidden_eeg, hidden_eog, hidden_emg])
        else:
            hidden_combined = tf.concat([hidden_eeg, hidden_eog, hidden_emg], 2)
        nHid = hidden_combined.get_shape()

        # Regularization

        if ac_config.is_training and (
                ac_config.keep_prob < 1.0):  # should this be ac_config.is_training + ac_config.keep_prob < 1.0
            iKeepProb = ac_config.keep_prob
            oKeepProb = ac_config.keep_prob
        else:
            iKeepProb = 1
            oKeepProb = 1

        # Layer hidden
        with tf.variable_scope('hidden_hidden') as scope:
            if ac_config.lstm:
                cell = tf.nn.rnn_cell.BasicLSTMCell(ac_config.num_hidden, forget_bias=1.0, state_is_tuple=True)
                cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=iKeepProb, output_keep_prob=oKeepProb)
                initial_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(self._initial_state[:, :ac_config.num_hidden],
                                                              self._initial_state[:, ac_config.num_hidden:])
                outputs, final_state = tf.compat.v1.nn.dynamic_rnn(cell, hidden_combined, dtype=tf.float32,
                                                         initial_state=initial_state)

            else:
                hidden_combined = tf.reshape(hidden_combined, [-1, int(nHid[2])])
                weights = sc_conv._variable_with_weight_decay('weights', shape=[nHid[2], ac_config.num_hidden],
                                                              stddev=0.04, wd=0.00001)
                biases = sc_conv._variable_on_cpu('biases', ac_config.num_hidden, tf.constant_initializer(0.01))
                outputs = tf.compat.v1.nn.relu(tf.add(tf.matmul(hidden_combined, weights), biases), name=scope.name)
                # sc_conv._activation_summary(outputs)

        # Layer out
        with tf.variable_scope('hidden_output') as scope:
            outputs = tf.reshape(outputs, [-1, ac_config.num_hidden])
            weights = sc_conv._variable_with_weight_decay('weights', shape=[ac_config.num_hidden, ac_config.num_classes],
                                                          stddev=0.04, wd=0.00001)
            biases = sc_conv._variable_on_cpu('biases', ac_config.num_classes, tf.constant_initializer(0.001))
            logits = tf.add(tf.matmul(outputs, weights), biases, name=scope.name)
            # sc_conv._activation_summary(logits)

        # Evaluate

        cross_ent = self.intelligent_cost(logits)
        loss = self.gather_loss()
        self._loss = loss
        self._logits = logits
        self._cross_ent = cross_ent
        self._softmax = tf.compat.v1.nn.softmax(logits)
        self._predict = tf.argmax(self._softmax, 1)
        self._correct = tf.equal(tf.argmax(logits, 1), tf.argmax(self._targets, 1))
        self._accuracy = tf.reduce_mean(tf.cast(self._correct, tf.float32))
        self._confidence = tf.reduce_sum(tf.multiply(self._softmax, self._targets), 1)
        self._baseline = (tf.reduce_mean(self._targets, 0))

        if ac_config.lstm:
            if tf.__version__ < '1.0':
                self._final_state = tf.concat(1, [final_state.c, final_state.h])
            else:
                self._final_state = tf.concat([final_state.c, final_state.h], 1)

        if not ac_config.is_training:
            return

        # Optimize
        optimizer = tf.train.MomentumOptimizer(learning_rate=self._learning_rate, momentum=0.9)
        variables_averages = tf.train.ExponentialMovingAverage(0.999)

        optimize = optimizer.minimize(self._loss)
        variables_averages_op = variables_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([optimize, variables_averages_op]):
            self._train_op = tf.no_op(name='train')

    def intelligent_cost(self, logits):
        logits = tf.clip_by_value(logits, -1e10, 1e+10)
        cross_ent = tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self._targets)
        # cross_ent = tf.mul(cross_ent, self._mask)
        cross_ent = tf.reduce_mean(cross_ent)  # / tf.reduce_sum(self._mask)
        tf.compat.v1.add_to_collection('losses', cross_ent)

        return cross_ent

    def gather_loss(self):

        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg_loss')
        losses = tf.compat.v1.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.

        # tf.scalar_summary(total_loss.op.name +' (raw)', total_loss)
        # tf.scalar_summary(total_loss.op.name, loss_averages.average(total_loss))

        return total_loss

    @property
    def features(self):
        return self._features

    @property
    def final_state(self):
        return self._final_state

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def targets(self):
        return self._targets

    @property
    def mask(self):
        return self._mask

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def cost(self):
        return self._cost

    @property
    def loss(self):
        return self._loss

    @property
    def cross_ent(self):
        return self._cross_ent

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def baseline(self):
        return self._baseline

    @property
    def train_op(self):
        return self._train_op

    @property
    def predict(self):
        return self._predict

    @property
    def logits(self):
        return self._logits

    @property
    def confidence(self):
        return self._confidence

    @property
    def ar_prob(self):
        return self._ar_prob

    @property
    def softmax(self):
        return self._softmax

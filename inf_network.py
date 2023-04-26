import inf_convolution as sc_conv

import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCModel(torch.nn.Module):
    def __init__(self, ac_config, initial_state: torch.Tensor):
        super(SCModel, self).__init__()

        self.model_name = ac_config.model_name
        self.initial_state = initial_state
        self.is_training: bool = ac_config.is_training
        self.n_features: int = ac_config.n_features
        self.n_hidden: int = ac_config.n_hidden
        self.n_classes: int = ac_config.n_classes

        self.segment_size: int = ac_config.segsize
        self.keep_prob: float = ac_config.keep_prob if self.is_training else 1.0
        self.is_lstm: bool = bool(ac_config.lstm)

        if self.is_lstm:
            self.cell = nn.LSTM(self.n_features, self.n_hidden)
            self.dropout = nn.Dropout(self.keep_prob)
        else:
            self.weights = nn.Parameter(
                nn.init.trunc_normal_(
                    torch.empty(self.n_features, self.n_hidden), std=0.04
                )
            )
            self.biases = nn.Parameter(torch.ones(self.n_hidden) * 0.01)

        self.out_weights = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.n_hidden, self.n_classes), std=0.04)
        )
        self.out_biases = nn.Parameter(torch.ones(self.n_classes) * 0.01)

    def forward(self, inputs: torch.Tensor):
        # Input Layer
        batch_size = inputs.size(dim=0)
        inputs = inputs.reshape([batch_size, -1, self.segment_size, self.n_features])

        hidden_eeg = sc_conv.main(
            inputs[:, :, :, :400],
            self.model_name,
            self.is_training,
            self.segment_size,
            "eeg",
            batch_size,
        )
        hidden_eog = sc_conv.main(
            inputs[:, :, :, 400:1600],
            self.model_name,
            self.is_training,
            self.segment_size,
            "eog",
            batch_size,
        )
        hidden_emg = sc_conv.main(
            inputs[:, :, :, 1600:],
            self.model_name,
            self.is_training,
            self.segment_size,
            "emg",
            batch_size,
        )

        hidden_combined = torch.cat((hidden_eeg, hidden_eog, hidden_emg), 2)

        # Hidden Layer
        if self.is_lstm:
            if self.initial_state is None:
                self.initial_state = (
                    torch.zeros(1, batch_size, self.n_hidden),
                    torch.zeros(1, batch_size, self.n_hidden),
                )
            outputs, final_state = self.cell(inputs)
        else:
            hidden_combined = hidden_combined.view(-1, hidden_combined.size(2))
            outputs = torch.matmul(hidden_combined, self.weights) + self.biases
            outputs = F.relu(outputs)

        # Output Layer
        outputs = outputs.view(-1, self.n_hidden)
        logits = torch.matmul(outputs, self.out_weights) + self.out_biases

        return logits

        # Evaluate

        cross_ent = self.intelligent_cost(logits)
        loss = self.gather_loss()
        self._loss = loss
        self._logits = logits
        self._cross_ent = cross_ent
        self._softmax = tf.nn.softmax(logits)
        self._predict = tf.argmax(self._softmax, 1)
        self._correct = tf.equal(tf.argmax(logits, 1), tf.argmax(self._targets, 1))
        self._accuracy = tf.reduce_mean(tf.cast(self._correct, tf.float32))
        self._confidence = tf.reduce_sum(tf.multiply(self._softmax, self._targets), 1)
        self._baseline = tf.reduce_mean(self._targets, 0)

        if ac_config.lstm:
            if tf.__version__ < "1.0":
                self._final_state = tf.concat(1, [final_state.c, final_state.h])
            else:
                self._final_state = tf.concat([final_state.c, final_state.h], 1)

        if not ac_config.is_training:
            return

        # Optimize
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=self._learning_rate, momentum=0.9
        )
        variables_averages = tf.train.ExponentialMovingAverage(0.999)

        optimize = optimizer.minimize(self._loss)
        variables_averages_op = variables_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([optimize, variables_averages_op]):
            self._train_op = tf.no_op(name="train")

    def intelligent_cost(self, logits):
        logits = tf.clip_by_value(logits, -1e10, 1e10)
        cross_ent = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self._targets
        )
        # cross_ent = tf.mul(cross_ent, self._mask)
        cross_ent = tf.reduce_mean(cross_ent)  # / tf.reduce_sum(self._mask)
        tf.add_to_collection("losses", cross_ent)

        return cross_ent

    def gather_loss(self):
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name="avg_loss")
        losses = tf.get_collection("losses")
        total_loss = tf.add_n(losses, name="total_loss")
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

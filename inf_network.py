import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import inf_convolution as sc_conv

# TODO: implement weight decay

class SCModel(torch.nn.Module):
    def __init__(self, ac_config, initial_state: torch.Tensor):
        super(SCModel, self).__init__()

        self.model_name: str = ac_config.model_name
        self.initial_state = initial_state
        # Should remove this later, but not sure how the conv layers use it
        self.is_training: bool = ac_config.is_training
        self.n_features: int = ac_config.n_features
        self.n_hidden: int = ac_config.n_hidden
        self.n_classes: int = ac_config.n_classes

        self.segment_size: int = ac_config.segsize
        self.dropout_rate: float = ac_config.keep_prob if self.is_training else 1.0
        self.is_lstm: bool = bool(ac_config.lstm)

        if self.is_lstm:
            self.cell = nn.LSTM(self.n_features, self.n_hidden)
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.fc_hidden = nn.Linear(self.n_features, self.n_hidden)
            nn.init.trunc_normal_(self.fc_hidden.weight, std=0.04)
            nn.init.constant_(self.fc_hidden.bias, 0.01)

        self.fc_out = nn.Linear(self.n_hidden, self.n_classes)
        nn.init.trunc_normal_(self.fc_out.weight, std=0.04)
        nn.init.constant_(self.fc_out.bias, 0.01)

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
            outputs, final_state = self.cell(hidden_combined)
            outputs = self.dropout(outputs)
        else:
            hidden_combined = hidden_combined.view(-1, hidden_combined.size(2))
            outputs = self.fc_hidden(hidden_combined)
            outputs = F.relu(outputs)

        # Output Layer
        outputs = outputs.view(-1, self.n_hidden)
        logits = self.fc_out(outputs)

        return logits

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


def train(
    model: SCModel,
    n_epochs: int,
    dataloader: DataLoader,
    lr: float = 0.001,
):
    model.train()

    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    for epoch in range(n_epochs):
        running_loss = 0.0
        total = correct = 0

        for features, labels in dataloader:
            optimizer.zero_grad()
            logits = model(features)
            loss = criteria(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            probs = F.softmax(logits)
            predictions = torch.argmax(probs, 1)

            running_loss += loss.item()
            total += features.size(0)
            correct += predictions.eq(labels).sum().item()
        print(
            f"Epoch {epoch}\n"
            f"Loss: {running_loss / len(dataloader)}\n"
            f"Acc: {correct / total}"
        )


def predict(model: SCModel, input):
    model.eval()
    logits = model(input)
    probs = F.softmax(logits)
    predictions = torch.argmax(probs, 1)
    return predictions
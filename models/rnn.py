from torch import nn as nn

from .base_pong_model import BasePongModel


class RNNModel(BasePongModel):
    def __init__(self, model_config):
        super(RNNModel, self).__init__(model_config)
        self.lstm = nn.LSTM(self.config.hidden_size, self.config.hidden_size, self.config.num_layers, batch_first=True, dropout=0.2)

    def _forward(self, x):
        out, _ = self.lstm(x)
        # Use the last output of the sequence for prediction
        last_out = out[:, -1, :]
        return last_out

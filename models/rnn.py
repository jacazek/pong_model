from torch import nn as nn

from .base_pong_model import BasePongModel
from . import ModelConfiguration

config = ModelConfiguration()


class RNNModel(BasePongModel):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, config.num_layers, batch_first=True, dropout=0.2)

    def _forward(self, x):
        out, _ = self.lstm(x)
        # Use the last output of the sequence for prediction
        last_out = out[:, -1, :]
        return last_out

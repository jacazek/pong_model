from abc import ABC, abstractmethod

import torch
from torch import nn as nn
import inject
from models import ModelConfiguration



class BasePongModel(nn.Module, ABC):
    @inject.params(config=ModelConfiguration)
    def __init__(self, config: ModelConfiguration):
        super(BasePongModel, self).__init__()
        self.config = config
        dropout_p = 0.4
        # Linear layer to expand input from 10 to 64 dimensions
        self.fc_feature_expansion = nn.Linear(config.input_size, config.hidden_size)

        # self.intermediate_head = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size * 4)
        # )

        self.regression_head = nn.Sequential(
            nn.Tanh(),
            nn.Dropout(dropout_p),
            nn.Linear(config.hidden_size, config.output_size)
        )
        self.classification_head = nn.Sequential(
            nn.Tanh(),
            nn.Dropout(dropout_p),
            nn.Linear(config.hidden_size, config.discrete_output_size),
        )

    @abstractmethod
    def _forward(self, x: torch.Tensor):
        pass

    def forward(self, x, discrete_temperature=1):
        x = self.fc_feature_expansion(x)
        # out, _ = self.lstm(x)
        # Use the last output of the sequence for prediction
        x = self._forward(x)
        # x = self.intermediate_head(x)
        regression_states = self.regression_head(x)  # / temperature
        classification_states = self.classification_head(x) / discrete_temperature
        return regression_states, classification_states

from collections import deque

import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import IterableDataset

from engine import EngineConfig
from pong_paddle import RandomPaddleFactory
from model_configuration import input_sequence_length, input_size, hidden_size, number_heads, num_layers, output_size, \
    discrete_output_size
from abc import ABC, abstractmethod


class PongDataset(IterableDataset):
    def __init__(self, state_generator, count):

        self.engine_config = EngineConfig()
        # increase max velocity of paddle to even out misses vs hits
        self.engine_config.set_paddle_factory(RandomPaddleFactory(max_velocity=0.009))
        self.generator = state_generator
        self.count = count

    def prepare_worker(self):
        worker_info = torch.utils.data.get_worker_info()
        # id = 0
        if worker_info is not None:
            self.count = int(self.count / worker_info.num_workers)
            # id = worker_info.id
        # print(f"worker {id} providing {self.count} samples")

    def generate(self):
        """
        Memory light loader, compute heavy
        """
        self.prepare_worker()
        window_size = input_sequence_length + 1
        window = deque(maxlen=window_size)
        for ball_data, paddle_data, collision_data, score_data in self.generator(engine_config=self.engine_config,
                                                                                 num_steps=self.count):
            window.append(ball_data + paddle_data + collision_data + score_data)
            if len(window) == window_size:
                # print(list(window))
                states = np.array(window)
                next_state = np.array(ball_data + collision_data + score_data)
                yield states[:input_sequence_length], next_state

    # def generate(self):
    #     """
    #     Memory heavy loader, compute light
    #     """
    #     self.prepare_worker()
    #     states = list(self.generator(self.count,
    #                                  engine_config=self.engine_config))
    #
    #     states = [(np.array([sum(item, []) for item in states[window_start:window_start + input_sequence_length]]),
    #                np.array(ball_data + collision_data + score_data)) for
    #               window_start, (ball_data, paddle_data, collision_data, score_data) in enumerate(states[input_sequence_length:])]
    #     for state in states:
    #         yield state

    def __iter__(self):
        return iter(self.generate())


class BasePongModel(nn.Module, ABC):
    def __init__(self):
        super(BasePongModel, self).__init__()
        # Linear layer to expand input from 10 to 64 dimensions
        self.fc_feature_expansion = nn.Linear(input_size, hidden_size)

        # self.intermediate_head = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size * 4)
        # )

        self.regression_head = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
        self.classification_head = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size, discrete_output_size),
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


class TransformerModel(BasePongModel):
    def __init__(self):
        super(TransformerModel, self).__init__()
        # Consider using decoder only with flash attention
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, hidden_size))

        # self.transformer = nn.TransformerEncoder(
        self.transformer_list = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=number_heads,
            dim_feedforward=hidden_size,
            batch_first=True,
        ) for _ in range(num_layers)])
        #     num_layers=num_layers,
        # )
        # self.transformer = nn.TransformerDecoderLayer(
        #     d_model=hidden_size,
        #     nhead=number_heads,
        #     dim_feedforward=hidden_size,
        #     # dropout=dropout
        # )

    def _forward(self, x):
        seq_len = x.size(1)
        positions = self.positional_encoding[:, :seq_len, :]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        x = x + positions
        # x = x.permute(1, 0, 2)  # Shape: (seq_len, batch_size, hidden_dim)
        for transformer in self.transformer_list:
            x = transformer(x, src_mask=causal_mask, is_causal=True)

        # Get the output of the transformer for each sequence by
        # Averaging over the sequence dimension to reduce dimensions for use in predicting next state
        # (similar to using last hidden state of LSTM)
        return x.mean(dim=1)


class RNNModel(BasePongModel):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

    def _forward(self, x):
        out, _ = self.lstm(x)
        # Use the last output of the sequence for prediction
        last_out = out[:, -1, :]
        return last_out

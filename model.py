from collections import deque

import numpy as np
from torch import nn as nn
from torch.utils.data import IterableDataset

from engine import EngineConfig, RandomPaddle
from model_configuration import input_sequence_length, input_size, hidden_size, number_heads, num_layers, output_size, \
    discrete_output_size


class PongDataset(IterableDataset):
    def __init__(self, state_generator, count):

        self.engine_config = EngineConfig()
        self.engine_config.paddle_class = RandomPaddle
        self.generator = state_generator
        self.count = count

    def generate(self):
        window_size = input_sequence_length + 1
        window = deque(maxlen=window_size)
        # for i in range(window_size):
        #     window.append([.5, .5] + [0.0 for i in range(6)])
        for ball_data, paddle_data, collision_data, score_data in self.generator(self.count,
                                                                                 engine_config=self.engine_config):
            window.append(ball_data + paddle_data + collision_data + score_data)
            if len(window) == window_size:
                states = np.array(window)
                next_state = np.array(ball_data + collision_data + score_data)
                yield states[:input_sequence_length], next_state

    def __iter__(self):
        return iter(self.generate())


class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()

        # Linear layer to expand input from 10 to 64 dimensions
        self.fc_feature_expansion = nn.Linear(input_size, hidden_size)

        # Transformer encoder layer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,  # Dimension of the input
                nhead=number_heads,  # Number of attention heads
                dim_feedforward=hidden_size,  # Feedforward dimension
            ),
            num_layers=num_layers  # Number of transformer layers
        )

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, output_size)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, discrete_output_size),
        )

    def forward(self, x):
        # x is of shape (batch_size, seq_len, input_dim)

        # Step 1: Expand the input to the hidden dimension (64)
        x = self.fc_feature_expansion(x)  # Shape: (batch_size, seq_len, hidden_dim)

        # Step 2: Permute to (seq_len, batch_size, hidden_dim) as expected by nn.TransformerEncoder
        x = x.permute(1, 0, 2)  # Shape: (seq_len, batch_size, hidden_dim)

        # Step 3: Pass through the transformer
        x = self.transformer(x)

        # Step 4: Get the output of the transformer for each sequence
        x = x.mean(dim=0)  # Average over the sequence dimension (optional)

        regression_states = self.regression_head(x)
        classification_states = self.classification_head(x)

        return regression_states, classification_states


class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()

        self.fc_feature_expansion = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

        self.regression_head = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, output_size)
        )
        self.classification_head = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size, discrete_output_size),
        )

    def forward(self, x):
        x = self.fc_feature_expansion(x)
        out, _ = self.lstm(x)
        # Use the last output of the sequence for prediction
        last_out = out[:, -1, :]
        regression_states = self.regression_head(last_out)
        classification_states = self.classification_head(last_out)
        return regression_states, classification_states

# class FCModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=2):
#         super(FCModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size * 4, num_layers, batch_first=True, dropout=0.2)
#         self.relu = nn.ReLU()
#         self.middle_fc = nn.Linear(hidden_size * 4, hidden_size)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         # LSTM output
#         out, _ = self.lstm(x)
#         # Use the last output of the sequence for prediction
#         last_out = out[:, -1, :]
#         return self.fc(self.middle_fc(self.relu(last_out)))

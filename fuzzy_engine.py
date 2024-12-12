from collections import deque

import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import IterableDataset
from engine import EngineConfig, generate_pong_states, Ball, RandomPaddle
from model_configuration import device, input_size, hidden_size, output_size, num_layers, model_path, input_sequence_length


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
        for item in self.generator(self.count, engine_config=self.engine_config):
            window.append(item[:4] + item[-6:])
            if len(window) == window_size:
                states = np.array(window)
                yield states[:input_sequence_length], states[-1][:4]



    def __iter__(self):
        return iter(self.generate())


class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc_position = nn.Linear(hidden_size, 2)
        self.fc_velocity = nn.Linear(hidden_size, 2)
        # self.relu = nn.ReLU()
        # self.middle_fc = nn.Linear(hidden_size, hidden_size)
        # self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # LSTM output
        out, _ = self.lstm(x)
        # Use the last output of the sequence for prediction
        last_out = out[:, -1, :]
        return torch.concat((self.fc_position(last_out), self.fc_velocity(last_out)), dim=1)

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




def generate_random_fuzzy_states(engine_config=EngineConfig(), num_steps=1000):
    ball = Ball()
    engine_config.ball = ball
    states = generate_pong_states(num_steps=num_steps, engine_config=engine_config)
    model = RNNModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    window_size = input_sequence_length
    window = deque(maxlen=window_size)
    counter = 0
    state = next(states)
    window.append(state[:4] + state[6:])
    for state in states:
        counter += 1
        # print(window[-1])
        fuzzy_state = model(torch.tensor([window]).float()).tolist()[0]
        window.append(fuzzy_state + state[6:10])
        options = [state, fuzzy_state + state[6:]]
        index = np.random.choice([0,1]) if counter % 100 == 0 else 0
        if (index == 1):
            ball.x = fuzzy_state[0]
            ball.y = fuzzy_state[1]
            ball.xv = fuzzy_state[2]
            ball.yv = fuzzy_state[3]
        yield options[index]

def generate_fuzzy_states(engine_config=EngineConfig(), num_steps=None):
    states = generate_pong_states(num_steps=num_steps, engine_config=engine_config)
    model = RNNModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    window_size = input_sequence_length
    window = deque(maxlen=window_size)
    counter = 0
    for i in range(window_size):
        state = next(states)
        window.append(state[:4] + state[6:])
    for state in states:
        # print(window[-1])
        fuzzy_state = model(torch.tensor([window]).float()).tolist()[0]
        window.append(fuzzy_state + state[6:])

        yield fuzzy_state + state[4:]


if __name__ == "__main__":
    engine_config = EngineConfig()
    for state in generate_fuzzy_states(engine_config, 10):
        print(state)



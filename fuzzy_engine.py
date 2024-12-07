from collections import deque

import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import IterableDataset
from engine import EngineConfig, finite_pong_state, Ball


class PongDataset(IterableDataset):
    def __init__(self, state_generator, count):

        self.generator = state_generator
        self.count = count

    def generate(self):
        window_size = 6
        window = deque(maxlen=6)
        # for i in range(window_size):
        #     window.append([.5, .5] + [0.0 for i in range(6)])
        for item in self.generator(self.count):
            window.append(item[:4] + item[-4:])
            if len(window) == window_size:
                states = np.array(window)
                yield states[:5], states[-1][:4]



    def __iter__(self):
        return iter(self.generate())


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size * 4, num_layers, batch_first=True, dropout=0.4)
        self.relu = nn.ReLU()
        self.middle_fc = nn.Linear(hidden_size * 4, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM output
        out, _ = self.lstm(x)
        # Use the last output of the sequence for prediction
        last_out = out[:, -1, :]
        return self.fc(self.middle_fc(self.relu(last_out)))




def generate_fuzzy_states(engine_config=EngineConfig(), num_steps=1000):
    ball = Ball()
    engine_config.ball = ball
    states = finite_pong_state(num_steps=num_steps, engine_config=engine_config)
    model = RNNModel(8, 8, 4, 2)
    model.load_state_dict(torch.load("pong_rnn_model.pth", weights_only=True))
    model.eval()
    window_size = 5
    window = deque(maxlen=5)
    counter = 0
    for i in range(window_size):
        window.append([0.0 for i in range(8)])
    for state in states:
        counter += 1
        # print(window[-1])
        fuzzy_state = model(torch.tensor([window]).float()).tolist()[0]
        window.append(fuzzy_state + state[6:])
        options = [state, fuzzy_state + state[4:]]
        index = np.random.choice([0,1]) if counter % 100 == 0 else 0
        if (index == 1):
            ball.x = fuzzy_state[0]
            ball.y = fuzzy_state[1]
            ball.xv = fuzzy_state[2]
            ball.yv = fuzzy_state[3]
        yield options[index]


if __name__ == "__main__":
    engine_config = EngineConfig()
    for state in generate_fuzzy_states(engine_config, 10):
        print(state)



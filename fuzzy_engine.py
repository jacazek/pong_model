from collections import deque

import torch
from engine import EngineConfig, generate_pong_states
from model_configuration import input_sequence_length, discrete_output_size, device
from runtime_configuration import Model, model_path, classification_threshold, temperature_variance
import numpy as np


# def generate_random_fuzzy_states(engine_config=EngineConfig(), num_steps=1000):
#     ball = Ball()
#     engine_config.ball = ball
#     states = generate_pong_states(num_steps=num_steps, engine_config=engine_config)
#     model = Model()
#     model.load_state_dict(torch.load(model_path, weights_only=True))
#     model.eval()
#     window_size = input_sequence_length
#     window = deque(maxlen=window_size)
#     counter = 0
#     state = next(states)
#     window.append(state[:4] + state[6:])
#     for state in states:
#         counter += 1
#         # print(window[-1])
#         fuzzy_state = model(torch.tensor([window]).float()).tolist()[0]
#         window.append(fuzzy_state + state[6:10])
#         options = [state, fuzzy_state + state[6:]]
#         index = np.random.choice([0,1]) if counter % 100 == 0 else 0
#         if (index == 1):
#             ball.x = fuzzy_state[0]
#             ball.y = fuzzy_state[1]
#             ball.xv = fuzzy_state[2]
#             ball.yv = fuzzy_state[3]
#         yield options[index]

def generate_fuzzy_states(engine_config=EngineConfig(), num_steps=None):
    states = generate_pong_states(num_steps=num_steps, engine_config=engine_config)
    model = Model().to(device=device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    window_size = input_sequence_length
    window = deque(maxlen=window_size)
    counter = 0
    for i in range(window_size):
        ball_data, paddle_data, collision_data, score_data = next(states)
        window.append(ball_data + paddle_data + collision_data + score_data)
    for _, paddle_data, _, _ in states:
        # print(window[-1])
        temperature = torch.from_numpy(
            np.random.uniform(1.0 - temperature_variance, 1.0 + temperature_variance, discrete_output_size)).to(
            device=device)
        ball_data, discrete_data = model(torch.tensor([window]).to(device=device, dtype=torch.float), temperature)
        ball_data = ball_data.tolist()[0]

        discrete_probabilities = torch.sigmoid(discrete_data)
        # print(discrete_probabilities)
        discrete_data = (discrete_probabilities > classification_threshold).int()
        discrete_data = discrete_data.tolist()[0]
        window.append(ball_data + paddle_data + discrete_data)
        yield ball_data, paddle_data, discrete_data[:4], discrete_data[4:]


if __name__ == "__main__":
    engine_config = EngineConfig()
    for state in generate_fuzzy_states(engine_config, 10):
        print(state)

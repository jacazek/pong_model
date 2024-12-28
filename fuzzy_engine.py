from collections import deque

import torch
from engine import EngineConfig, Field
from exact_engine import generate_pong_states
from model_configuration import input_sequence_length, discrete_output_size, input_size, device
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


def generate_fuzzy_states(engine_config=EngineConfig(), num_steps=None,):
    state_generator = _generate_fuzzy_states(engine_config)
    if num_steps is None:
        for state in state_generator:
            yield state
    else:
        for step in range(num_steps):
            yield next(state_generator)
def _generate_fuzzy_states(engine_config=EngineConfig()):
    dt = 1  # Time step

    paddle_width = engine_config.paddle_width_percent / engine_config.field_width * engine_config.field_width
    paddle_height = engine_config.paddle_height_percent / engine_config.field_height * engine_config.field_height
    # Initialize ball position and velocity
    field = Field(engine_config.field_width, engine_config.field_height)

    # convert to using factory?
    left_paddle = engine_config.paddle_factory.create_paddle(paddle_width, paddle_height, field.left,
                                                             0 - paddle_height / 2, field)
    right_paddle = engine_config.paddle_factory.create_paddle(paddle_width, paddle_height, field.right - paddle_width,
                                                              0 - paddle_height / 2, field)

    model = Model().to(device=device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device(device)))
    model.eval()
    window_size = input_sequence_length
    window = deque(maxlen=window_size)
    window.extend(np.zeros((input_sequence_length, input_size), dtype=float))
    while True: # for _, paddle_data, _, _ in states:
        left_paddle.update(dt)
        right_paddle.update(dt)
        paddle_data = left_paddle.vectorize_state() + right_paddle.vectorize_state()
        # print(window[-1])
        temperature = torch.from_numpy(
            np.random.uniform(1.0 - temperature_variance, 1.0 + temperature_variance, discrete_output_size)).to(
            device=device)
        # print(temperature.tolist())
        ball_data, discrete_data = model(torch.tensor([window]).to(device=device, dtype=torch.float), temperature)
        ball_data = ball_data.tolist()[0]
        discrete_probabilities = torch.sigmoid(discrete_data)

        # print(discrete_data.tolist())
        # print(discrete_probabilities.tolist())
        # print(discrete_probabilities)
        classes = (discrete_probabilities > classification_threshold).int()
        classes = classes.tolist()[0]
        # print(classes)
        window.append(ball_data + paddle_data + classes)
        yield ball_data, paddle_data, classes[:4], classes[4:]


if __name__ == "__main__":
    engine_config = EngineConfig()
    for state in generate_fuzzy_states(engine_config, 10):
        print(state)

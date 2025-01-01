from collections import deque

import torch
from engine import EngineConfig, Field
from exact_engine import generate_pong_states
from models import ModelConfiguration
from runtime_configuration import Model, model_path, classification_threshold, temperature_variance
import numpy as np

config = ModelConfiguration()

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

    left_paddle = engine_config.paddle_factory.create_paddle(paddle_width, paddle_height, field.left,
                                                             0 - paddle_height / 2, field)
    right_paddle = engine_config.paddle_factory.create_paddle(paddle_width, paddle_height, field.right - paddle_width,
                                                              0 - paddle_height / 2, field)

    model = Model().to(device=config.device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device(config.device)))
    model.eval()
    window_size = config.input_sequence_length
    window = deque(maxlen=window_size)
    window.extend(np.zeros((config.input_sequence_length, config.input_size), dtype=float))
    while True:
        left_paddle.update(dt)
        right_paddle.update(dt)
        paddle_data = left_paddle.vectorize_state() + right_paddle.vectorize_state()
        temperature = torch.from_numpy(
            np.random.uniform(1.0 - temperature_variance, 1.0 + temperature_variance, config.discrete_output_size)).to(
            device=config.device)
        ball_data, discrete_data = model(torch.tensor(np.array([window])).to(device=config.device, dtype=torch.float), temperature)
        ball_data = ball_data.tolist()[0]
        discrete_probabilities = torch.sigmoid(discrete_data)

        classes = (discrete_probabilities > classification_threshold).int()
        classes = classes.tolist()[0]
        window.append(ball_data + paddle_data + classes)
        yield ball_data, paddle_data, classes[:4], classes[4:]


if __name__ == "__main__":
    engine_config = EngineConfig()
    for state in generate_fuzzy_states(engine_config, 10):
        print(state)

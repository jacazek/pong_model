from collections import deque

import inject
import torch
import requests
import mlflow

from game.state import State
from main_arguments import MainArguments
from runtime_configuration import classification_threshold, temperature_variance, mlflow_server_url
import numpy as np
from model_loaders import load_model

# Consider creating a context manager to configure the server from which models are loaded and use the manager
# within load_model to avoid leaking mlflow details higher than where is needed
mlflow.set_tracking_uri(mlflow_server_url)
try:
    response = requests.get(mlflow_server_url)
    if response.status_code == 200:
        mlflow.set_tracking_uri(mlflow_server_url)
        print(f"MLflow Tracking URL set to: {mlflow_server_url}")
    else:
        print(f"MLflow server at {mlflow_server_url} is not available. Status code: {response.status_code}")
        print("Will load models from local mlruns directory")
except requests.exceptions.RequestException as e:
    print(f"Failed to connect to MLflow server at {mlflow_server_url}. Error: {e}")
    print("Will load models from local mlruns directory")

# config = ModelConfiguration()

def generate_fuzzy_states(num_steps=None):
    state_generator = _generate_fuzzy_states()
    if num_steps is None:
        for state in state_generator:
            yield state
    else:
        for step in range(num_steps):
            yield next(state_generator)

@inject.params(game_state=State, main_arguments=MainArguments)
def _generate_fuzzy_states(game_state=State, main_arguments=MainArguments):
    dt = 1  # Time step

    model = load_model(main_arguments.model_path)


    model.eval()
    window_size = main_arguments.input_sequence_length
    window = deque(maxlen=window_size)
    window.extend(np.zeros((main_arguments.input_sequence_length, main_arguments.input_size), dtype=float))
    while True:
        game_state.left_paddle.update(dt)
        game_state.right_paddle.update(dt)
        paddle_data = game_state.left_paddle.vectorize_state() + game_state.right_paddle.vectorize_state()
        # temperature = torch.from_numpy(
        #     np.random.uniform(1.0 - temperature_variance, 1.0 + temperature_variance, main_arguments.discrete_output_size) * 100).to(
        #     device=main_arguments.device)
        temperature = 1 # larger temperature is more creativw
        ball_data, discrete_data = model(torch.tensor(np.array([window])).to(device=main_arguments.device, dtype=torch.float), temperature)

        ball_data = ball_data.tolist()[0]
        discrete_probabilities = torch.sigmoid(discrete_data)

        classes = (discrete_probabilities > classification_threshold).int()
        classes = classes.tolist()[0]

        window.append(ball_data + paddle_data + classes)
        yield ball_data, paddle_data, classes[:4], classes[4:]


if __name__ == "__main__":
    for state in generate_fuzzy_states(10):
        print(state)

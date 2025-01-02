from collections import deque

import numpy as np
import torch
from torch.utils.data import IterableDataset

from game.configuration import EngineConfig
from game.paddle import RandomPaddleFactory
from . import ModelConfiguration
import inject

config = ModelConfiguration()


class PongDataset(IterableDataset):
    generator = inject.attr("generator")
    def __init__(self, count):
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
        window_size = config.input_sequence_length + 1
        window = deque(maxlen=window_size)
        for ball_data, paddle_data, collision_data, score_data in self.generator(num_steps=self.count):
            window.append(ball_data + paddle_data + collision_data + score_data)
            if len(window) == window_size:
                # print(list(window))
                states = np.array(window)
                next_state = np.array(ball_data + collision_data + score_data)
                yield states[:config.input_sequence_length], next_state

    # def generate(self):
    #     """
    #     Memory heavy loader, compute light
    #     """
    #     self.prepare_worker()
    #     states = list(self.generator(num_steps=self.count))
    #
    #     states = [(np.array([sum(item, []) for item in states[window_start:window_start + config.input_sequence_length]]),
    #                np.array(ball_data + collision_data + score_data)) for
    #               window_start, (ball_data, paddle_data, collision_data, score_data) in enumerate(states[config.input_sequence_length:])]
    #     for state in states:
    #         yield state

    def __iter__(self):
        return iter(self.generate())

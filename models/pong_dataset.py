from collections import deque

import numpy as np
import torch
from torch.utils.data import IterableDataset

from game.configuration import EngineConfig
from game.paddle import RandomPaddleFactory
from . import ModelConfiguration

config = ModelConfiguration()


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
        window_size = config.input_sequence_length + 1
        window = deque(maxlen=window_size)
        for ball_data, paddle_data, collision_data, score_data in self.generator(engine_config=self.engine_config,
                                                                                 num_steps=self.count):
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

from abc import ABC, abstractmethod

import numpy as np
import pygame
import inject

from game.configuration import EngineConfig
from game.field import Field


class Paddle(ABC):
    field = inject.attr(Field)
    engine_config = inject.attr(EngineConfig)

    def __init__(self, width, height, x, y, yv):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.yv = yv

    @abstractmethod
    def update(self, dt):
        pass

    def vectorize_state(self):
        return [self.x, self.y, self.yv]


class UserPaddle(Paddle):
    def __init__(self, width, height, x, y, yv, up_key, down_key):
        Paddle.__init__(self, width, height, x, y, yv)
        self.up_key = up_key
        self.down_key = down_key

    def update(self, dt):
        self.yv = 0
        keys = pygame.key.get_pressed()
        if keys[self.up_key]:
            if self.y > self.field.top:
                self.yv = -.01 # Move up
        elif keys[self.down_key]:
            if self.y + self.height < self.field.bottom:
                self.yv = .01  # Move down

        # if assigned key is pressed, move paddle in indicated direction at configured velocity until collision with box
        self.y += self.yv * dt


class RandomPaddle(Paddle):
    def __init__(self, width, height, x, y, min_velocity=0.025, max_velocity=0.1):
        Paddle.__init__(self, width, height, x, y, 0)
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.count = 0
        self.random_ng = np.random.default_rng()
        self.yv = self.random_paddle_velocity()

    def random_paddle_velocity(self):
        return self.random_ng.uniform(-1*self.max_velocity, self.max_velocity)

    def update(self, dt):
        self.count += 1
        self.y += self.yv * dt
        if self.y <= self.field.top or self.y + self.height >= self.field.bottom:
            self.yv *= -1  # Reverse vertical velocity
        else:
            self.yv = self.random_paddle_velocity() if self.count % 100 == 0 else self.yv


class PaddleFactory(ABC):
    engine_config = inject.attr(EngineConfig)
    field = inject.attr(Field)

    def get_paddle_dimensions(self):
        return self.engine_config.paddle_width_percent / self.field.width * self.field.width, self.engine_config.paddle_height_percent / self.field.height * self.field.height

    @abstractmethod
    def create_left_paddle(self):
        pass

    @abstractmethod
    def create_right_paddle(self):
        pass


class UserPaddleFactory(PaddleFactory):
    def __init__(self):
        super().__init__()

    def create_left_paddle(self):
        width, height = self.get_paddle_dimensions()
        return UserPaddle(width, height, self.field.left, 0 - height / 2, 0, pygame.K_q, pygame.K_a)

    def create_right_paddle(self):
        width, height = self.get_paddle_dimensions()
        return UserPaddle(width, height, self.field.right - width,
                                                                  0 - height / 2, 0, pygame.K_UP, pygame.K_DOWN)

class RandomPaddleFactory(PaddleFactory):
    def create_left_paddle(self):
        width, height = self.get_paddle_dimensions()
        return RandomPaddle(width, height, self.field.left, 0 - height / 2, min_velocity=self.min_velocity, max_velocity=self.max_velocity)

    def create_right_paddle(self):
        width, height = self.get_paddle_dimensions()
        return RandomPaddle(width, height, self.field.right - width,
                                                                  0 - height / 2, min_velocity=self.min_velocity, max_velocity=self.max_velocity)

    def __init__(self, min_velocity=0.025, max_velocity=0.1):
        super().__init__()
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity

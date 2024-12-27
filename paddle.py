from abc import ABC, abstractmethod
import numpy as np
import pygame


class Paddle(ABC):
    def __init__(self, width, height, x, y, yv, field):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.yv = yv
        self.field = field

    @abstractmethod
    def update(self, dt):
        pass

    def vectorize_state(self):
        return [self.x, self.y, self.yv]


class UserPaddle(Paddle):
    def __init__(self, width, height, x, y, yv, field, up_key, down_key):
        Paddle.__init__(self, width, height, x, y, yv, field)
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
    def __init__(self, width, height, x, y, field, min_velocity=0.025, max_velocity=0.1):
        Paddle.__init__(self, width, height, x, y, 0, field)
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.yv = self.random_paddle_velocity()
        self.count = 0

    def random_paddle_velocity(self):
        return np.random.choice([np.random.uniform(self.min_velocity, self.max_velocity),
                                 np.random.uniform(self.max_velocity * -1, self.min_velocity * -1)])

    def update(self, dt):
        self.count += 1
        self.y += self.yv * dt
        if self.y <= self.field.top or self.y + self.height >= self.field.bottom:
            self.yv *= -1  # Reverse vertical velocity
        else:
            self.yv = self.random_paddle_velocity() if self.count % 100 == 0 else self.yv


class PaddleFactory(ABC):
    @abstractmethod
    def create_paddle(self, width, height, x, y, field):
        pass


class UserPaddleFactory(PaddleFactory):
    def __init__(self):
        self.count = 0

    def create_paddle(self, width, height, x, y, field):
        if self.count % 2 == 0:
            self.count += 1
            return UserPaddle(width, height, x, y, 0, field, pygame.K_q, pygame.K_a)
        if self.count %2 == 1:
            self.count += 1
            return UserPaddle(width, height, x, y, 0, field, pygame.K_UP, pygame.K_DOWN)

class RandomPaddleFactory(PaddleFactory):
    def create_paddle(self, width, height, x, y, field):
        return RandomPaddle(width, height, x, y, field, min_velocity=self.min_velocity, max_velocity=self.max_velocity)

    def __init__(self, min_velocity=0.025, max_velocity=0.1):
        PaddleFactory.__init__(self)
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity

from abc import ABC, abstractmethod
import numpy as np


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
    def __init__(self, width, height, x, y, yv, field):
        Paddle.__init__(self, width, height, x, y, yv, field)

    def update(self, dt):
        # if assigned key is pressed, move paddle in indicated direction at configured velocity until collision with box
        pass


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
        if self.y <= 0 or self.y + self.height >= self.field.height:
            self.yv *= -1  # Reverse vertical velocity
        else:
            self.yv = self.random_paddle_velocity() if self.count % 100 == 0 else self.yv


class PaddleFactory(ABC):
    @abstractmethod
    def create_paddle(self, width, height, x, y, field):
        pass


class UserPaddleFactory(ABC):
    def create_paddle(self, width, height, x, y, field):
        pass


class RandomPaddleFactory(PaddleFactory):
    def create_paddle(self, width, height, x, y, field):
        return RandomPaddle(width, height, x, y, field, min_velocity=self.min_velocity, max_velocity=self.max_velocity)

    def __init__(self, min_velocity=0.025, max_velocity=0.1):
        PaddleFactory.__init__(self)
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity

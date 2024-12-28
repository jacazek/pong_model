import numpy as np

from paddle import UserPaddleFactory, Paddle


# def random_velocity(max=0.025, min=0.005):
#     # return np.random.choice([np.random.uniform(min, max), np.random.uniform(max * -1, min * -1)])
#     random_ng = np.random.default_rng()
#     [x, y] = random_ng.uniform(max * -1, max, 2)
#     return x, y

def random_velocity_generator(min=0.001, max=0.025):
    # count = 0
    random_ng = np.random.default_rng()
    while True:
        x = random_ng.uniform(max*-1, min*-1) if random_ng.choice([True, False]) else random_ng.uniform(min, max)
        y = random_ng.uniform(max*-1, max)
        # random_x = np.random.uniform(min, max)
        # random_y = random_velocity(min=min, max=max)
        # count += 1
        yield x, y


class EngineConfig:
    def __init__(self, ball_radius_percent=.01, field_width=1.0, field_height=1.0, paddle_width_percent=.01,
                 paddle_height_percent=.2, ball=None):
        self.ball_radius_percent = ball_radius_percent
        self.paddle_width_percent = paddle_width_percent
        self.paddle_height_percent = paddle_height_percent
        self.field_width = field_width
        self.field_height = field_height
        self.ball = ball
        self.paddle_factory = UserPaddleFactory

    def set_paddle_factory(self, paddle_factory):
        self.paddle_factory = paddle_factory



class Field:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.left = width / 2.0 * -1
        self.right = width / 2.0
        self.top = height / 2.0 * -1
        self.bottom = height / 2.0


class Ball:
    def __init__(self, initial_x=0, initial_y=0, initial_xv=0, initial_yv=0, radius=1, left_paddle: Paddle = None,
                 right_paddle: Paddle = None,
                 field: Field = None):
        self.x = initial_x
        self.y = initial_y
        self.xv = initial_xv
        self.yv = initial_yv
        self.radius = radius
        self.left_paddle = left_paddle
        self.right_paddle = right_paddle
        self.field = field

    def reset(self, x, y, xv, yv):
        self.x = x
        self.y = y
        self.xv = xv
        self.yv = yv

    def update(self, dt):
        self.x += self.xv * dt
        self.y += self.yv * dt
        left_paddle_collision = 0
        right_paddle_collision = 0
        top_field_collision = 0
        bottom_field_collision = 0
        if self.y - self.radius <= self.field.top:
            self.yv *= -1
            top_field_collision = 1

        if self.y + self.radius >= self.field.bottom:
            self.yv *= -1  # Reverse vertical velocity
            bottom_field_collision = 1

        # Check for paddle collisions
        if self.x - self.radius < self.left_paddle.x + self.left_paddle.width and self.xv <= 0:  # Left paddle
            if self.left_paddle.y <= self.y <= self.left_paddle.y + self.left_paddle.height:
                self.xv *= -1  # Reverse horizontal velocity
                left_paddle_collision = 1

        if self.x + self.radius > self.right_paddle.x and self.xv > 0:  # Right paddle
            if self.right_paddle.y <= self.y <= self.right_paddle.y + self.right_paddle.height:
                self.xv *= -1  # Reverse horizontal velocity
                right_paddle_collision = 1

        return [left_paddle_collision, right_paddle_collision, top_field_collision, bottom_field_collision]



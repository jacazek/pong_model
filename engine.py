import numpy as np


def random_velocity(max=0.05, min=0.01):
    return np.random.choice([np.random.uniform(min, max), np.random.uniform(max * -1, min * -1)])


class EngineConfig:
    def __init__(self, ball_radius_percent=.02, field_width=1, field_height=1, paddle_width_percent=.01,
                 paddle_height_percent=.2, ball=None):
        self.ball_radius_percent = ball_radius_percent
        self.paddle_width_percent = paddle_width_percent
        self.paddle_height_percent = paddle_height_percent
        self.field_width = field_width
        self.field_height = field_height
        self.ball = ball


class Field:
    def __init__(self, width, height):
        self.width = width
        self.height = height


class Paddle:
    def __init__(self, width, height, x, y, yv, field):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.yv = yv
        self.field = field
        self.count = 0

    def update(self, dt):
        self.count += 1
        self.y += self.yv * dt
        if self.y <= 0 or self.y + self.height >= self.field.height:
            self.yv *= -1  # Reverse vertical velocity
        else:
            self.yv = random_velocity() if self.count % 100 == 0 else self.yv


class Ball():
    def __init__(self, initial_x=0, initial_y=0, initial_xv=0, initial_yv=0, radius=1, left_paddle: Paddle=None, right_paddle: Paddle=None,
                 field: Field=None):
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
        if self.y -self.radius <= 0:
            self.yv *= -1
            top_field_collision = 1

        if self.y + self.radius >= self.field.height:
            self.yv *= -1  # Reverse vertical velocity
            bottom_field_collision = 1

        # Check for paddle collisions
        if self.x - self.radius <= self.left_paddle.width and self.xv <= 0:  # Left paddle
            if self.left_paddle.y <= self.y <= self.left_paddle.y + self.left_paddle.height:
                self.xv *= -1  # Reverse horizontal velocity
                left_paddle_collision = 1

        if self.x + self.radius >= self.right_paddle.x and self.xv > 0:  # Right paddle
            if self.right_paddle.y <= self.y <= self.right_paddle.y + self.right_paddle.height:
                self.xv *= -1  # Reverse horizontal velocity
                right_paddle_collision = 1

        return [left_paddle_collision, right_paddle_collision, top_field_collision, bottom_field_collision]


def finite_pong_state(num_steps=1000, engine_config=EngineConfig()):
    state_generator = generate_pong_states(engine_config)
    for step in range(num_steps):
        yield next(state_generator)


def generate_pong_states(engine_config=EngineConfig()):
    # states = []  # To store ball position, velocity, and paddle positions

    dt = 1  # Time step
    score_1 = 0
    score_2 = 0
    blocked_1 = 0
    blocked_2 = 0

    paddle_width = engine_config.paddle_width_percent * engine_config.field_width
    paddle_height = engine_config.paddle_height_percent * engine_config.field_height

    # Initialize ball position and velocity
    field = Field(engine_config.field_width, engine_config.field_height)
    left_paddle = Paddle(width=paddle_width, height=paddle_height, x=0, y=field.height / 2, yv=random_velocity(),
                         field=field)
    right_paddle = Paddle(width=paddle_width, height=paddle_height, x=field.width - paddle_width, y=field.height / 2,
                          yv=random_velocity(), field=field)
    ball = engine_config.ball or Ball()
    ball.reset(0.5, 0.5, random_velocity(0.03), random_velocity(0.03))
    ball.left_paddle=left_paddle
    ball.right_paddle=right_paddle
    ball.field=field
    ball.radius=engine_config.ball_radius_percent * engine_config.field_height

    # Initialize paddle positions

    # Save the current state
    yield [ball.x, ball.y, ball.xv, ball.yv, left_paddle.y, right_paddle.y, score_1, score_2, 0, 0, blocked_1, blocked_2]

    while True:
        # Update ball position
        left_paddle.update(dt)
        right_paddle.update(dt)
        collisions = ball.update(dt)

        # Reset if ball goes out of bounds (optional)
        if ball.x + ball.radius < 0 and not collisions[0]:
            score_2 += 1
            ball.reset(.5, .5, random_velocity(0.03), random_velocity(0.03))
        if ball.x - ball.radius > field.width and not collisions[1]:
            score_1 += 1
            ball.reset(.5, .5, random_velocity(0.03), random_velocity(0.03))

        blocked_1 += collisions[0]
        blocked_2 += collisions[1]

        # Update paddle positions (static or random movement for simulation)
        # Here, paddles are static, but you can add logic for movement.
        # Update paddle position

        # Save the current state
        yield [ball.x, ball.y, ball.xv, ball.yv, left_paddle.y, right_paddle.y, score_1, score_2, collisions[0],
               collisions[1], blocked_1, blocked_2]

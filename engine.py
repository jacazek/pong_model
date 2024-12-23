import numpy as np


def random_velocity(max=0.05, min=0.01):
    return np.random.choice([np.random.uniform(min, max), np.random.uniform(max * -1, min * -1)])

def random_velocity_generator(min=0.01, max=0.05):
    count = 0
    while True:
        random_x = np.random.uniform(min, max)
        random_y = random_velocity(min=min, max=max)
        count += 1
        yield (random_x if count % 2 == 0 else random_x * -1, random_y)


class EngineConfig:
    def __init__(self, ball_radius_percent=.02, field_width=1, field_height=1, paddle_width_percent=.01,
                 paddle_height_percent=.2, ball=None):
        self.ball_radius_percent = ball_radius_percent
        self.paddle_width_percent = paddle_width_percent
        self.paddle_height_percent = paddle_height_percent
        self.field_width = field_width
        self.field_height = field_height
        self.ball = ball
        self.paddle_class = Paddle
    def set_paddle_class(self, PaddleClass):
        self.paddle_class = PaddleClass


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
        # if assigned key is pressed, move paddle in indicated direction at configured velocity until collision with box
        pass


class RandomPaddle(Paddle):
    def __init__(self, width, height, x, y, yv, field):
        Paddle.__init__(self, width, height, x, y, yv, field)
        self.yv = RandomPaddle.random_paddle_velocity()

    @staticmethod
    def random_paddle_velocity():
        return random_velocity(min=.08, max=.1)

    def update(self, dt):
        self.count += 1
        self.y += self.yv * dt
        if self.y <= 0 or self.y + self.height >= self.field.height:
            self.yv *= -1  # Reverse vertical velocity
        else:
            self.yv = RandomPaddle.random_paddle_velocity() if self.count % 100 == 0 else self.yv


class Ball:
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


def generate_pong_states(num_steps=None, engine_config=EngineConfig()):
    state_generator = _generate_pong_states(engine_config)
    if num_steps is None:
        for state in state_generator:
            yield state
    else:
        for step in range(num_steps):
            yield next(state_generator)


def _generate_pong_states(engine_config=EngineConfig()):
    # states = []  # To store ball position, velocity, and paddle positions

    dt = 1  # Time step
    ball_random_velocity = random_velocity_generator(min=0.01)

    paddle_width = engine_config.paddle_width_percent * engine_config.field_width
    paddle_height = engine_config.paddle_height_percent * engine_config.field_height

    # Initialize ball position and velocity
    field = Field(engine_config.field_width, engine_config.field_height)

    # convert to using factory?
    left_paddle = engine_config.paddle_class(paddle_width, paddle_height, 0, field.height / 2, random_velocity(),
                         field)
    right_paddle = engine_config.paddle_class(paddle_width, paddle_height, field.width - paddle_width, field.height / 2,
                          random_velocity(), field)
    ball = engine_config.ball or Ball()
    x, y = next(ball_random_velocity)
    ball.reset(0.5, 0.5, x,y)
    ball.left_paddle=left_paddle
    ball.right_paddle=right_paddle
    ball.field=field
    ball.radius=engine_config.ball_radius_percent * engine_config.field_height

    ball_data = [ball.x, ball.y, ball.xv, ball.yv]
    paddle_data = [left_paddle.x, left_paddle.y, left_paddle.yv, right_paddle.x, right_paddle.y, right_paddle.yv]
    collision_data = [0, 0, 0, 0] # with what did the ball collide?
    score_data = [0, 0] # was a score made?


    # Save the current state
    yield ball_data, paddle_data, collision_data, score_data
    # yield [ball.x, ball.y, ball.xv, ball.yv, left_paddle.y, right_paddle.y, score_1, score_2, 0, 0, blocked_1, blocked_2]

    while True:
        score_data = [0, 0]
        # Update ball position
        left_paddle.update(dt)
        right_paddle.update(dt)
        collisions = ball.update(dt)

        # Reset if ball goes out of bounds (optional)
        if ball.x + ball.radius < 0 and not collisions[0]:
            score_data[1] = 1 # right team scored
            x, y = next(ball_random_velocity)
            ball.reset(.5, .5,  x, y)
        if ball.x - ball.radius > field.width and not collisions[1]:
            score_data[0] = 1 # left team scored

            x, y = next(ball_random_velocity)
            ball.reset(.5, .5, x, y)

        # Update paddle positions (static or random movement for simulation)
        # Here, paddles are static, but you can add logic for movement.
        # Update paddle position

        # Save the current state
        ball_data = [ball.x, ball.y, ball.xv, ball.yv]
        paddle_data = [left_paddle.x, left_paddle.y, left_paddle.yv, right_paddle.x, right_paddle.y, right_paddle.yv]
        collision_data = collisions  # with what did the ball collide?
        yield ball_data, paddle_data, collision_data, score_data
        # yield [ball.x, ball.y, ball.xv, ball.yv, left_paddle.y, right_paddle.y, score_1, score_2, collisions[0],
        #        collisions[1], blocked_1, blocked_2]

from engine import EngineConfig, random_velocity_generator, Field, Ball


def generate_pong_states(engine_config=EngineConfig(), num_steps=None):
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
    ball_random_velocity = random_velocity_generator()

    paddle_width = engine_config.paddle_width_percent /engine_config.field_width * engine_config.field_width
    paddle_height = engine_config.paddle_height_percent / engine_config.field_height * engine_config.field_height
    # print(f"{paddle_width}, {paddle_height}")
    # Initialize ball position and velocity
    field = Field(engine_config.field_width, engine_config.field_height)

    # convert to using factory?
    left_paddle = engine_config.paddle_factory.create_paddle(paddle_width, paddle_height, field.left, 0 - paddle_height/2, field)
    right_paddle = engine_config.paddle_factory.create_paddle(paddle_width, paddle_height, field.right - paddle_width,
                                                              0 - paddle_height/2, field)
    ball = engine_config.ball or Ball()
    x, y = next(ball_random_velocity)
    ball.reset(0, 0, x, y)
    ball.left_paddle = left_paddle
    ball.right_paddle = right_paddle
    ball.field = field
    ball.radius = engine_config.ball_radius_percent * engine_config.field_height

    ball_data = [ball.x, ball.y, ball.xv, ball.yv]
    paddle_data = left_paddle.vectorize_state() + right_paddle.vectorize_state()
    collision_data = [0, 0, 0, 0]  # with what did the ball collide?
    score_data = [0, 0]  # was a score made?

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
        if ball.x + ball.radius < field.left and not collisions[0]:
            score_data[1] = 1  # right team scored
            x, y = next(ball_random_velocity)
            ball.reset(0, 0, x, y)
        if ball.x - ball.radius > field.right and not collisions[1]:
            score_data[0] = 1  # left team scored

            x, y = next(ball_random_velocity)
            ball.reset(0, 0, x, y)

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

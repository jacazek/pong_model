from game.state import State
import inject
import numpy as np

def random_velocity_generator(min=0.005, max=0.025):
    # count = 0
    random_ng = np.random.default_rng()
    while True:
        x = random_ng.uniform(max*-1, min*-1) if random_ng.choice([True, False]) else random_ng.uniform(min, max)
        y = random_ng.uniform(max*-1, max)
        # random_x = np.random.uniform(min, max)
        # random_y = random_velocity(min=min, max=max)
        # count += 1
        yield x, y


def generate_pong_states(num_steps=None):
    state_generator = _generate_pong_states()
    if num_steps is None:
        for state in state_generator:
            yield state
    else:
        for step in range(num_steps):
            yield next(state_generator)


@inject.params(game_state=State)
def _generate_pong_states(game_state: State = None):
    dt = 1  # Time step
    ball_random_velocity = random_velocity_generator()

    left_paddle = game_state.left_paddle
    right_paddle = game_state.right_paddle
    ball = game_state.ball
    field = game_state.field

    x, y = next(ball_random_velocity)
    ball.reset(0, 0, x, y)

    ball_data = [ball.x, ball.y, ball.xv, ball.yv]
    paddle_data = left_paddle.vectorize_state() + right_paddle.vectorize_state()
    collision_data = [0, 0, 0, 0]  # with what did the ball collide?
    score_data = [0, 0]  # was a score made?

    # Save the current state
    yield ball_data, paddle_data, collision_data, score_data

    while True:
        score_data = [0, 0]
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

        # Save the current state
        ball_data = [ball.x, ball.y, ball.xv, ball.yv]
        paddle_data = [left_paddle.x, left_paddle.y, left_paddle.yv, right_paddle.x, right_paddle.y, right_paddle.yv]
        collision_data = collisions  # with what did the ball collide?
        yield ball_data, paddle_data, collision_data, score_data


from engine import random_velocity_generator
from game.state import State
import inject


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

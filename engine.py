import numpy as np

def random_velocity(max=.01):
    return np.random.uniform(-max, max)

class EngineConfig:
    def __init__(self, ball_radius_percent = .01, field_width = 1, field_height = 1, paddle_width_percent = .01, paddle_height_percent = .2):
        self.ball_radius_percent = ball_radius_percent
        self.paddle_width_percent = paddle_width_percent
        self.paddle_height_percent = paddle_height_percent
        self.field_width = field_width
        self.field_height = field_height


def finite_pong_state(num_steps=1000, engine_config=EngineConfig()):
    state_generator = generate_pong_states(engine_config)
    for step in range(num_steps):
        yield next(state_generator)


def generate_pong_states(engine_config=EngineConfig()):
    # states = []  # To store ball position, velocity, and paddle positions

    dt = 1  # Time step
    score_1 = 0
    score_2 = 0

    ball_radius = engine_config.ball_radius_percent * engine_config.field_height
    paddle_width = engine_config.paddle_width_percent * engine_config.field_width
    paddle_height = engine_config.paddle_height_percent * engine_config.field_height

    # Initialize ball position and velocity
    ball_x = np.random.uniform(ball_radius, engine_config.field_width - ball_radius)
    ball_y = np.random.uniform(ball_radius, engine_config.field_height - ball_radius)
    ball_vx = random_velocity()  # Random initial velocity
    ball_vy = random_velocity()

    # Initialize paddle positions
    paddle1_y = engine_config.field_height / 2
    paddle2_y = engine_config.field_height / 2
    paddle1_vy = random_velocity(.05)
    paddle2_vy = random_velocity(.05)

    # Save the current state
    yield [ball_x, ball_y, ball_vx, ball_vy, paddle1_y, paddle2_y, score_1, score_2, 0, 0]

    while True:
        paddle1_collision = 0
        paddle2_collision = 0
        # Update ball position
        ball_x += ball_vx * dt
        ball_y += ball_vy * dt


        # Check for wall collisions (top and bottom)
        if ball_y - ball_radius <= 0 or ball_y + ball_radius >= engine_config.field_height:
            ball_vy *= -1  # Reverse vertical velocity

        # Check for paddle collisions
        if ball_x - ball_radius <= paddle_width and ball_vx <= 0:  # Left paddle
            if paddle1_y <= ball_y <= paddle1_y + paddle_height:
                ball_vx *= -1  # Reverse horizontal velocity
                paddle1_collision = 1

        if ball_x + ball_radius >= engine_config.field_width - paddle_width  and ball_vx > 0:  # Right paddle
            if paddle2_y <= ball_y <= paddle2_y + paddle_height:
                ball_vx *= -1  # Reverse horizontal velocity
                paddle2_collision = 1

        # Reset if ball goes out of bounds (optional)
        if ball_x < 0:
            score_2 += 1
        if ball_x > engine_config.field_width:
            score_1 += 1

        if ball_x < 0 or ball_x > engine_config.field_width:
            # random ball start location
            ball_x = .5 # np.random.uniform(ball_radius, engine_config.field_width - ball_radius)
            ball_y = .5 # np.random.uniform(ball_radius, engine_config.field_height - ball_radius)
            ball_vx = random_velocity()
            ball_vy = random_velocity()

        # Update paddle positions (static or random movement for simulation)
        # Here, paddles are static, but you can add logic for movement.
        # Update paddle position
        paddle1_y += paddle1_vy * dt
        paddle2_y += paddle2_vy * dt

        paddle_half_height = paddle_height / 2
        if paddle1_y <= 0 or paddle1_y + paddle_height >= engine_config.field_height:
            paddle1_vy *= -1  # Reverse vertical velocity

        if paddle2_y <= 0 or paddle2_y + paddle_height >= engine_config.field_height:
            paddle2_vy *= -1  # Reverse vertical velocity

        # Save the current state
        yield [ball_x, ball_y, ball_vx, ball_vy, paddle1_y, paddle2_y, score_1, score_2, paddle1_collision, paddle2_collision]


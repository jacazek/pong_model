class EngineConfig:
    def __init__(self, ball_radius_percent=.01, paddle_width_percent=.01,
                 paddle_height_percent=.2, min_ball_velocity=.005, max_ball_velocity=.025):
        self.ball_radius_percent = ball_radius_percent
        self.paddle_width_percent = paddle_width_percent
        self.paddle_height_percent = paddle_height_percent
        self.max_ball_velocity = max_ball_velocity
        self.min_ball_velocity = min_ball_velocity


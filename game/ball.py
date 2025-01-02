import inject

from game.paddle import Paddle
from game.field import Field
from game.configuration import EngineConfig

class Ball:
    engin_config = inject.attr(EngineConfig)
    field: Field = inject.attr(Field)
    left_paddle: Paddle = inject.attr("left_paddle")
    right_paddle: Paddle = inject.attr("right_paddle")

    def __init__(self, initial_x=0, initial_y=0, initial_xv=0, initial_yv=0):
        self.x = initial_x
        self.y = initial_y
        self.xv = initial_xv
        self.yv = initial_yv
        self.radius = self.engin_config.ball_radius_percent * self.field.height

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

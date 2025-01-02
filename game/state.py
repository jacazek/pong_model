import inject
from game.ball import Ball
from game.configuration import EngineConfig
from game.field import Field
from game.paddle import PaddleFactory

class State:
    ball = inject.attr(Ball)
    field = inject.attr(Field)
    engineConfig = inject.attr(EngineConfig)
    paddleFactory = inject.attr(PaddleFactory)

    def __init__(self):
        self.left_paddle = self.paddleFactory.create_left_paddle()
        self.right_paddle = self.paddleFactory.create_right_paddle()
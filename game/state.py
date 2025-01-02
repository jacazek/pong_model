import inject
from game.ball import Ball
from game.configuration import EngineConfig
from game.field import Field

class State:
    ball = inject.attr(Ball)
    field = inject.attr(Field)
    engineConfig = inject.attr(EngineConfig)
    left_paddle = inject.attr("left_paddle")
    right_paddle = inject.attr("right_paddle")
import inject
from game.ball import Ball
from game.configuration import EngineConfig
from game.field import Field
from game.score import Score

class State:
    ball = inject.attr(Ball)
    field = inject.attr(Field)
    engineConfig = inject.attr(EngineConfig)
    left_paddle = inject.attr("left_paddle")
    right_paddle = inject.attr("right_paddle")
    scores = inject.attr(Score)
"""
Given a balls initial position, direction and
"""
from game.configuration import EngineConfig
from game.field import Field
from exact_engine import generate_pong_states
from game.paddle import UserPaddleFactory, RandomPaddleFactory, PaddleFactory
from game.score import Score
from game.state import State
from game.ball import Ball
from fuzzy_engine import generate_fuzzy_states
import pygame
import inject
from models import ModelConfiguration
from main_arguments import MainArguments

# Initialize Pygame
pygame.init()

# Game parameters
screen_width = 800
screen_height = 400
screen_quarter = int(screen_width/4)
background_color = (0, 0, 0)  # Black
ball_color = (255, 255, 255)  # White
paddle_color = (255, 255, 255)  # White
paddle_color_collision = (255, 0, 0)  # White

half_screen_width = screen_width / 2.0
half_screen_height = screen_height / 2.0

# Initialize Pygame screen
screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
pygame.display.set_caption("Pong State Renderer")


def translate(point):
    x,y = point
    return x + half_screen_width, y + half_screen_height

@inject.params(field=Field)
def scale_to_screen(point, field: Field = None):
    x, y = point
    x = (x + field.width / 2.0) / field.width
    y = (y + field.height / 2.0) / field.height
    return int(x * screen_width), int(y * screen_height)


font_size = 24
font = pygame.font.Font(None, font_size) # None = default font
small_font = pygame.font.Font(None, 18)
text_color = (255, 255, 255)  # White

@inject.params(scores=Score)
def update_scores(state, scores: Score = None):
    ball_data, paddle_data, collision_data, score_data = state
    scores.update(*(score_data+collision_data[:2]))

@inject.params(scores=Score)
def render_scores(scores: Score = None):
    score1_surface = font.render(f"{scores.left_score} | {scores.left_blocked}", True, text_color)
    score2_surface = font.render(f"{scores.right_score} | {scores.right_blocked}", True, text_color)

    screen.blit(score1_surface, (screen_quarter - score1_surface.get_width()/2, 10))
    screen.blit(score2_surface, (screen_width - screen_quarter - score2_surface.get_width()/2, 10))

def render_field():
    pygame.draw.line(screen, text_color, (screen_width/2, 0), (screen_width/2, screen_height), 1)

# Function to render the state
@inject.params(engine_config=EngineConfig, field=Field)
def render_state(state, count, engine_config: EngineConfig = None, field: Field = None):

    ball_data, paddle_data, collision_data, score_data = state
    ball_x, ball_y, _, _ = ball_data
    paddle1_x, paddle1_y, paddle1_vy, paddle2_x, paddle2_y, paddle2_vy = paddle_data
    collision_1, collision_2, collision_3, collision_4 = collision_data

    # Clear the screen
    screen.fill(background_color)

    # Draw the ball
    pygame.draw.circle(screen, ball_color, scale_to_screen((ball_x, ball_y)), engine_config.ball_radius_percent*(screen_width / field.width), 0)

    # Draw the paddles... might be able to use data from paddles directly?
    paddle_width = engine_config.paddle_width_percent / field.width * screen_width
    paddle_height = engine_config.paddle_height_percent / field.height * screen_height

    pygame.draw.rect(screen, paddle_color_collision if collision_1 else paddle_color, scale_to_screen((paddle1_x, paddle1_y)) + (paddle_width, paddle_height))  # Left paddle
    pygame.draw.rect(screen, paddle_color_collision if collision_2 else paddle_color, scale_to_screen((paddle2_x, paddle2_y)) + (paddle_width, paddle_height))  # Right paddle

    render_scores()

    render_field()

    debug_surface = small_font.render(f"{count}", True, (0, 255, 0))
    screen.blit(debug_surface, (0, 10))

    # Update the display
    pygame.display.flip()


# Main loop to render the state

@inject.params(main_arguments=MainArguments)
def main(main_arguments: MainArguments):
    global screen, screen_width, screen_height
    running = True
    for index, state in enumerate(main_arguments.generator()):
        if not running:
            break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.VIDEORESIZE:  # Handle window resizing
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                screen_width = event.w
                screen_height = event.h


        update_scores(state)

        # Render the state
        render_state(state, index)

        # Add a delay to control the frame rate
        pygame.time.delay(30)

    # Quit Pygame
    pygame.quit()

def configure_main(binder: inject.Binder):
    main_arguments = MainArguments.get_arguments()
    binder.bind(MainArguments, main_arguments)
    binder.bind(ModelConfiguration, main_arguments)
    # immediatly construct and bind an instance to the given key
    binder.bind(Field, Field(1.0, 1.0))
    binder.bind(EngineConfig, EngineConfig())
    binder.bind(PaddleFactory, UserPaddleFactory())
    binder.bind(Score, Score())

    # defer constructions for objects with more complex dependencies
    # what are needed during initialization
    # will create singleton instance upon retrieval of the object bound to the key
    # necessary as trying to access instances during bind configuration will crash with injector not configured error
    binder.bind_to_constructor("left_paddle", lambda: inject.instance(PaddleFactory).create_left_paddle())
    binder.bind_to_constructor("right_paddle", lambda: inject.instance(PaddleFactory).create_right_paddle())
    binder.bind_to_constructor(Ball, Ball)
    binder.bind_to_constructor(State, State)


if __name__ == "__main__":
    inject.configure(configure_main)
    main()

"""
Given a balls initial position, direction and
"""
from engine import generate_pong_states, EngineConfig
from paddle import RandomPaddleFactory
from fuzzy_engine import generate_fuzzy_states

print("Hello world!")

import pygame
import numpy as np

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


def scale_to_screen(x, y):
    return int(x * screen_width), int(y * screen_height)


engine_config = EngineConfig()
engine_config.set_paddle_factory(RandomPaddleFactory(max_velocity=0.05))

score_1 = 0
score_2 = 0
blocked_1 = 0
blocked_2 = 0

font_size = 24
font = pygame.font.Font(None, font_size) # None = default font
small_font = pygame.font.Font(None, 18)
text_color = (255, 255, 255)  # White

def update_scores(state):
    global blocked_1, blocked_2, score_1, score_2
    ball_data, paddle_data, collision_data, score_data = state
    blocked_1 += collision_data[0]
    blocked_2 += collision_data[1]
    score_1 += score_data[0]
    score_2 += score_data[1]

# Function to render the state
def render_state(state, count):

    ball_data, paddle_data, collision_data, score_data = state
    # print(collision_data)
    # print(score_data)
    ball_x, ball_y, _, _ = ball_data
    paddle1_x, paddle1_y, paddle1_vy, paddle2_x, paddle2_y, paddle2_vy = paddle_data
    collision_1, collision_2, collision_3, collision_4 = collision_data

    # Clear the screen
    screen.fill(background_color)

    # Draw the ball
    pygame.draw.circle(screen, ball_color, scale_to_screen(ball_x, ball_y), engine_config.ball_radius_percent*screen_width)

    # Draw the paddles
    paddle_width = engine_config.paddle_width_percent * screen_width
    paddle_height = engine_config.paddle_height_percent * screen_height

    pygame.draw.rect(screen, paddle_color_collision if collision_1 else paddle_color, (0, paddle1_y * screen_height, paddle_width, paddle_height))  # Left paddle
    pygame.draw.rect(screen, paddle_color_collision if collision_2 else paddle_color, (screen_width - paddle_width, paddle2_y *  screen_height, paddle_width, paddle_height))  # Right paddle

    render_scores(score_1, score_2, blocked_1, blocked_2)

    render_field()

    debug_surface = small_font.render(f"{count}", True, (0, 255, 0))
    screen.blit(debug_surface, (0, 10))

    # Update the display
    pygame.display.flip()


def render_scores(score1, score2, blocked1, blocked2):
    score1_surface = font.render(f"{score1} | {blocked1}", True, text_color)
    score2_surface = font.render(f"{score2} | {blocked2}", True, text_color)

    screen.blit(score1_surface, (screen_quarter - score1_surface.get_width()/2, 10))
    screen.blit(score2_surface, (screen_width - screen_quarter - score2_surface.get_width()/2, 10))

def render_field():
    pygame.draw.line(screen, text_color, (screen_width/2, 0), (screen_width/2, screen_height), 1)

# Initialize Pygame screen
screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
pygame.display.set_caption("Pong State Renderer")

# Main loop to render the state
running = True

for index, state in enumerate(generate_fuzzy_states(engine_config)):
# for index, state in enumerate(generate_pong_states(num_steps=50000, engine_config=engine_config)):
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

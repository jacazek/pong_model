"""
Given a balls initial position, direction and
"""
from engine import generate_pong_states, EngineConfig
from fuzzy_engine import generate_fuzzy_states

print("Hello world!")

import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Game parameters
screen_width = 800
screen_height = 400
background_color = (0, 0, 0)  # Black
ball_color = (255, 255, 255)  # White
paddle_color = (255, 255, 255)  # White


def scale_to_screen(x, y):
    return int(x * screen_width), int(y * screen_height)


# Load a state (example state)
# state = [400, 200, 5, -5, 170, 230]  # [ball_x, ball_y, ball_vx, ball_vy, paddle1_y, paddle2_y]

engine_config = EngineConfig()

# Function to render the state
def render_state(state):

    ball_x, ball_y, _, _, paddle1_y, paddle2_y, score_1, score_2, _, _ = state

    # Clear the screen
    screen.fill(background_color)

    # Draw the ball
    pygame.draw.circle(screen, ball_color, scale_to_screen(ball_x, ball_y), engine_config.ball_radius_percent*screen_height)

    # Draw the paddles
    paddle_width = engine_config.paddle_width_percent * screen_width
    paddle_height = engine_config.paddle_height_percent * screen_height

    pygame.draw.rect(screen, paddle_color, (0, paddle1_y * screen_height, paddle_width, paddle_height))  # Left paddle
    pygame.draw.rect(screen, paddle_color, (screen_width - paddle_width, paddle2_y *  screen_height, paddle_width, paddle_height))  # Right paddle

    render_scores(score_1, score_2)

    # Update the display
    pygame.display.flip()

font_size = 36
font = pygame.font.Font(None, font_size)  # None = default font
text_color = (255, 255, 255)  # White
def render_scores(score1, score2):
    score1_surface = font.render(f"{score1} | {score2}", True, text_color)
    # score2_surface = font.render(str(int(score2)), True, text_color)
    screen.blit(score1_surface, (screen_width/2 - score1_surface.get_width()/2, 10))
    # screen.blit(score2_surface, (screen_width - 60, 10))


# Initialize Pygame screen
screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
pygame.display.set_caption("Pong State Renderer")

# Main loop to render the state
running = True
# states = generate_pong_states(2000)
for state in generate_fuzzy_states(engine_config):
# for state in generate_pong_states(engine_config):
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

    # Render the state
    render_state(state)

    # Add a delay to control the frame rate
    pygame.time.delay(30)

# Quit Pygame
pygame.quit()
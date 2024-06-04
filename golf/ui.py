import pygame
from constants import *

def draw_button(screen, button_rect, text, font, button_color=(200, 200, 200), text_color=BLACK):
    pygame.draw.rect(screen, button_color, button_rect)
    text_surf = font.render(text, True, text_color)
    text_rect = text_surf.get_rect(center=button_rect.center)
    screen.blit(text_surf, text_rect)

def draw_club_selection(screen, font, current_club):
    text = f"Current Club: {current_club}"
    text_surf = font.render(text, True, BLACK)
    screen.blit(text_surf, (10, 760))

def draw_score_tracker(screen, font, score):
    text = f"Score: {score}"
    text_surf = font.render(text, True, BLACK)
    screen.blit(text_surf, (600 - text_surf.get_width() // 2, 10))

def draw_lie_shower(screen, font, lie):
    text = f"Lie: {lie}"
    text_surf = font.render(text, True, BLACK)
    screen.blit(text_surf, (600 - text_surf.get_width() // 2, 50))

def draw_ui(screen, font, button_rect, current_club, score, lie):
    draw_button(screen, button_rect, "Start New Game", font)
    draw_club_selection(screen, font, current_club)
    draw_score_tracker(screen, font, score)
    draw_lie_shower(screen, font, lie)
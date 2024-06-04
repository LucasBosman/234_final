import pygame
from course import GolfCourse
from ball import Ball
from ui import draw_button, draw_club_selection, draw_score_tracker
from aiming import AimingSystem
from game import Game
from constants import *


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Random Golf Course Generator")
    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()

    button_rect = pygame.Rect((SCREEN_WIDTH - BUTTON_WIDTH) // 2, SCREEN_HEIGHT - BUTTON_HEIGHT - 20, BUTTON_WIDTH, BUTTON_HEIGHT)

    game = Game(screen, font, button_rect, "profile.json")

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        screen.fill(WHITE)
        game.draw(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            game.handle_event(event, mouse_pos)

        pygame.display.flip()
        clock.tick(120)

    pygame.quit()

if __name__ == "__main__":
    main()

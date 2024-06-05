import pygame
from course import GolfCourse
from ball import Ball
import json
from ui import draw_ui, draw_out_of_bounds, draw_hole_complete
from aiming import AimingSystem
from constants import *


class Game:
    def __init__(self, screen, font, button_rect, profile_file):
        self.screen = screen
        self.font = font
        self.button_rect = button_rect
        self.load_profile(profile_file)
        self.reset_game()

    def reset_game(self):
        self.course = GolfCourse(par=4, difficulty=2)
        start_pos = self.course.teebox.rotated_rect.center
        self.ball = Ball(start_pos[0], start_pos[1], 3, WHITE)
        self.aiming_system = AimingSystem(params=self.profile)
        self.score = 0
        self.done = False
        self.current_club_index = 0
        self.current_lie = 'Teebox'

    def load_profile(self, profile_file):
        with open(profile_file, 'r') as file:
            self.profile = json.load(file)
        
        self.clubs = list(self.profile.keys())

    def draw(self, mouse_pos):
        self.course.draw(self.screen)
        self.ball.draw(self.screen)

        draw_ui(self.screen, self.font, self.button_rect, self.aiming_system.current_club, self.score, self.current_lie)

        if self.done:
            draw_hole_complete(self.screen, self.font)
            return # return early before we draw the arrow and gaussian

        if self.ball.target_pos is None:
            self.aiming_system.draw_arrow(self.screen, self.ball.get_pos(), mouse_pos)
            self.aiming_system.draw_gaussian(self.screen, self.ball.get_pos(), mouse_pos)
        else:
            self.aiming_system.draw_arrow(self.screen, self.ball.get_pos(), self.ball.locked_mouse_pos)
            self.aiming_system.draw_gaussian(self.screen, self.ball.get_pos(), self.ball.locked_mouse_pos)

    def handle_event(self, event, mouse_pos):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # check if the button was clicked
            if self.button_rect.collidepoint(event.pos):
                self.reset_game()
            else:
                # if the game is done, don't allow the player to take any more shots
                if self.done:
                    return

                target_pos = self.aiming_system.sample_gaussian(self.ball.get_pos(), mouse_pos)

                # check for out of bounds or water hazard
                next_lie = self.course.get_element_at(target_pos.astype(int))
                if next_lie == 'Out of Bounds' or next_lie == 'Water Hazard':
                    self.handle_out_of_bounds(target_pos, next_lie)

                # check if the ball is in the hole (green)
                elif next_lie == 'Green':
                    # animate the ball to the green
                    self.ball.start_animation(target_pos)
                    self.ball.animate_path(self.screen, pygame.time.Clock(), self.course, self.aiming_system, self.button_rect, self.font, self.score, self.current_lie)
                    self.score += 1
                    self.current_lie = self.course.get_element_at(self.ball.get_pos().astype(int))
                    self.done = True
                
                # regular shot
                else:
                    # animate the ball to the target position
                    self.ball.start_animation(target_pos)
                    self.ball.animate_path(self.screen, pygame.time.Clock(), self.course, self.aiming_system, self.button_rect, self.font, self.score, self.current_lie)
                    self.score += 1
                    self.current_lie = self.course.get_element_at(self.ball.get_pos().astype(int))
                    self.aiming_system.set_lie(self.current_lie)
        
        elif event.type == pygame.KEYDOWN:
            # if the game is done, don't allow the player to switch clubs
            if self.done:
                return
            if event.key == pygame.K_LEFT:
                self.current_club_index = (self.current_club_index - 1) % len(self.clubs)
                self.aiming_system.change_club(self.clubs[self.current_club_index])
            elif event.key == pygame.K_RIGHT:
                self.current_club_index = (self.current_club_index + 1) % len(self.clubs)
                self.aiming_system.change_club(self.clubs[self.current_club_index])

    def handle_out_of_bounds(self, target_pos, next_lie):
        # animate the ball to the OB position, then show flashing text that says "OUT OF BOUNDS!", then increment the score by 2, then go back to the previous ball location
        previous_pos = self.ball.get_pos() # save the previous position
        self.ball.start_animation(target_pos)
        self.ball.animate_path(self.screen, pygame.time.Clock(), self.course, self.aiming_system, self.button_rect, self.font, self.score, self.current_lie)
        self.score += 2
        self.current_lie = next_lie
        draw_out_of_bounds(self.screen, self.font)
        # wait
        pygame.display.flip()
        pygame.time.wait(1000)
        # go back to the previous position
        self.ball.move_to(*previous_pos)
        self.current_lie = self.course.get_element_at(self.ball.get_pos().astype(int))
        return

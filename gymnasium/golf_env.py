# ------------------------------------------------------------------------------------
# File: golf_env.py
# Authors: Guinness
# Date: 06/04/2024
# Description: This file contains the implementation of the GolfGameEnv class, which is a custom Gymnasium environment
# -------------------------------------------------------------------------------------

# import functions and classes
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ..golf.game import Game
from ..golf.constants import SCREEN_WIDTH, SCREEN_HEIGHT

class GolfGameEnv(gym.Env):
    def __init__(self, player_profile, course_profile):
        super().__init__()
        self.game = None
        # save player and course profiles
        self.player_profile = player_profile
        self.course_profile = course_profile

        # define action and observation spaces and reward range
        self.action_space = spaces.Dict({
            "club": spaces.Discrete(14),
            "direction": spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32),
        })
        self. observation_space = spaces.Dict({
            "ball_position": spaces.Box(low=0, high=SCREEN_WIDTH, shape=(2,), dtype=np.float32),
            "lie": spaces.Discrete(4),
            "course": spaces.Box(low=0, high=255, shape=(SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=np.uint8),
        })
        self.reward_range = (0, np.inf)
         
    def reset(self, seed=None):
        super().reset(seed=seed)
        #initialize game
        self.game = Game(screen, "golf/profile.json")

    def step(self, action):
        club, direction = action["club"], action["direction"]
        # set up the aiming system with the selected club and lie
        self.game.aiming_system.set_club(club)
        current_lie = self.game.course.get_element_at(self.game.ball.get_pos())
        self.game.aiming_system.set_lie(current_lie)

        # compute a target position based on the direction, where the magnitute is arbitrary
        target_pos = np.array([np.cos(direction), np.sin(direction)]) * 100

        # sample the next position of the ball
        next_pos = self.game.aiming_system.sample_gaussian(self.game.ball.get_pos(), target_pos)
        next_lie = self.game.course.get_element_at(next_pos)

        # handle out of bounds and water hazards
        if next_lie == "Out of Bounds" or next_lie == "Water Hazard":
            # update game state
            self.game.score += 2
            # set the variables which will be returned by the step function
            reward = -2
            terminated = False
        # handle hole completion (end of the episode)
        elif next_lie == "Green":
            # update game state
            self.game.score += 1
            self.game.ball.move_to(*next_pos)
            # set the variables which will be returned by the step function
            reward = -1
            terminated = True
        else:
            # update game state
            self.game.score += 1
            self.game.ball.move_to(*next_pos)
            # set the variables which will be returned by the step function
            reward = -1
            terminated = False

        # construct the observation object
        observation = {
            "ball_position": self.game.ball.get_pos(),
            "lie": next_lie,
            "course": pygame.surfarray(self.game.course.course_surface)
        }

        return observation, reward, terminated 
        
    def render(self):
        #render the game
        pass

    
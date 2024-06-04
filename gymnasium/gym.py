import gymnasium as gym
from golf_game import GolfGameEnv

#need to register environment first
gym.register(
    id='GolfGame-v0',
    entr_point='golf_game:GolfGameEnv',
    kwargs={'player': None}
)

#create player or sample player from previously made players
player = 

#Test env
env = gym.make('GolfGame-v0')
obs = env.reset()
env.render()


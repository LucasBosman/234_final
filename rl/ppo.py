import gymnasium as gym
import random
import torch
import numpy as np
from agent import PPOAgent
import ActorCnn CriticCnn


#need to register environment first
gym.register(
    id='GolfGame-v0',
    entr_point='golf_game:GolfGameEnv',
    kwargs={'player': None}
)

#make the environment
env = gym.make('GolfGame-v0')

#get device
device = torch.device("cuda" if torch.cuda.is_avaiable() else "cpu")
print("Device: ", device)

#reset env
env.reset()

IMAGE_SHAPE = (3, 1200, 800) # this is the size of our image... I added 3 here for RGB, but honestly IDK what input dim should be
GAMMA = 0.99 # this is discount factor
ALPHA = 0.0001 # actor LR
BETA = 0.0001 # critic LR
TAU = 0.95 # for use in calculating PPO
BATCH_SIZE = 32
PPO_EPOCH = 5 # number of epochs for each call to model learn
CLIP_PARAM = 0.2 # hyperparam for PPO
UPDATE_EVERY = 1000 # how often we update network

# Create our agent
agent = PPOAgent(IMAGE_SHAPE, device, GAMMA, ALPHA, BETA, TAU, UPDATE_EVERY, BATCH_SIZE, PPO_EPOCH, CLIP_PARAM, ActorCnn, CriticCnn)


def train(n_episodes=1000):
    for i_episode in range(1, n_episodes + 1): # Train for 1000 episodes
        state = env.reset() # Reset, aka start at beginning
        score = 0 #Set score to 0
        done = False
        while True:
            action, log_prob, value = agent.act(state) # Get predicted action and value, plus log_prob
            next_state, reward, done = env.step(action) # Get next state, reward, and whether or not we are done from our Simulator
            score += reward # Add reward to score
            agent.step(state, action, value, log_prob, reward, done, next_state) # Take a step on the agent (may or may not update model)
            
            if done:
                break
            else:
                state = next_state

        print(f"Episode {i_episode} achieved a score of {score}")

train()

# # TO VISUALIZE
# score = 0
# state = env.reset()
# done = False
# while True:
#     env.render()
#     action, _, _ = agent.act(state)
#     next_state, reward, done = env.step(action)
#     score += reward
#     state = next_state
#     if done:
#         print(f"Final score was {score}")
#         break
# env.close()

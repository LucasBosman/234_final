import gymnasium as gym
import random
import torch
import numpy as np
from agent import PPOAgent
import ActorCnn
#need to register environment first
gym.register(
    id='GolfGame-v0',
    entr_point='golf_game:GolfGameEnv',
    kwargs={'player': None}
)

env = gym.make('GolfGame-v0')

#get device
device = torch.device("cuda" if torch.cuda.is_avaiable() else "cpu")
print("Device: ", device)


env.reset()

IMAGE_SHAPE = (3, 1200, 800) #this is the size of our image
GAMMA = 0.99
ALPHA = 0.0001
BETA = 0.0001
TAU = 0.95
BATCH_SIZE = 32
PPO_EPOCH = 5
CLIP_PARAM = 0.2
UPDATE_EVERY = 1000 #how often we update network

agent = PPOAgent(IMAGE_SHAPE, SEED, device, GAMMA, ALPHA, BETA, TAU, UPDATE_EVERY, BATCH_SIZE, PPO_EPOCH, CLIP_PARAM, ActorCnn)


def train(n_episodes=1000):
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        done = False
        while True:
            action, log_prob, value = agent.act(state)
            next_state, reward, done = env.step(action)
            score += reward
            agent.step(state, action, value, log_prob, reward, done, next_state)
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

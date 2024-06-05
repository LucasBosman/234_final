import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

class PPOAgent():
    def __init__(self, image_shape, device, gamma, alpha, beta, tau, update_every, batch_size, ppo_epoch, clip_param, actor_m, critic_m):
        self.image_shape = image_shape
        self.seed = random.seed(0)
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.ppo_epoch = ppo_epoch
        self.clip_param = clip_param

        # Actor-Network
        self.actor_net = actor_m(image_shape, (1, 2), (1,)).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha)

        # Critic-Network
        self.critic_net = critic_m(image_shape).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.beta)

        # Memory
        self.log_probs = []
        self.values    = []
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.masks     = []
        self.entropies = []

        self.t_step = 0

    def step(self, state, action, value, log_prob, reward, done, next_state):
        # Unpack state tuple
        img, pos, lie = state
        next_img, next_pos, next_lie = next_state

        # Convert state components to tensors
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        pos = torch.tensor(pos, dtype=torch.float32).unsqueeze(0).to(self.device)
        lie = torch.tensor(lie, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Convert next_state components to tensors
        next_img = torch.from_numpy(next_img).unsqueeze(0).to(self.device)
        next_pos = torch.tensor(next_pos, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_lie = torch.tensor(next_lie, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Save experience in memory
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.states.append((img, pos, lie))
        self.rewards.append(torch.tensor(reward, dtype=torch.float32).to(self.device))
        self.actions.append(torch.tensor(action).to(self.device))
        self.masks.append(torch.tensor(1 - done, dtype=torch.float32).to(self.device))

        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            self.learn((next_img, next_pos, next_lie))
            self.reset_memory()
                
    def act(self, state):
        """Returns action, log_prob, value for given state as per current policy."""
        
        image = torch.from_numpy(state[0]).unsqueeze(0).to(self.device)
        position = torch.from_numpy(state[1]).unsqueeze(0).to(self.device)
        lie = torch.tensor([state[2]]).unsqueeze(0).to(self.device)
        state = (image, position, lie) # stitch together state

        theta, club_dist = self.actor_net(state) # get theta and club distribution
        value = self.critic_net(state) # get predicted value

        club = club_dist.sample()
        action = (theta.item(), club.item() + 1) # Add 1 to club to get a number from 1-14
        log_prob = club_dist.log_prob(club)

        return action, log_prob, value

    def learn(self, next_state):
        next_img, next_pos, next_lie = next_state
        next_value = self.critic_net((next_img.unsqueeze(0), next_pos.unsqueeze(0), next_lie.unsqueeze(0)))

        returns = torch.cat(self.compute_gae(next_value)).detach()
        self.log_probs = torch.cat(self.log_probs).detach()
        self.values = torch.cat(self.values).detach()
        self.states = [tuple(torch.cat(s) for s in zip(*self.states))]
        self.actions = torch.cat(self.actions)
        advantages = returns - self.values

        for _ in range(self.ppo_epoch):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(returns, advantages):

                theta, club_dist = self.actor_net(state)
                value = self.critic_net(state)

                entropy = club_dist.entropy().mean()
                new_log_probs = club_dist.log_prob(action[:, 1])

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

                # Minimize the loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.reset_memory()

    
    def ppo_iter(self, returns, advantage):
        memory_size = len(self.states)
        for _ in range(memory_size // self.batch_size):
            rand_ids = np.random.randint(0, memory_size, self.batch_size)
            yield self.states[rand_ids, :], self.actions[rand_ids], self.log_probs[rand_ids], returns[rand_ids, :], advantage[rand_ids, :]

    def reset_memory(self):
        self.log_probs = []
        self.values    = []
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.masks     = []
        self.entropies = []

    def compute_gae(self, next_value):
        gae = 0
        returns = []
        values = self.values + [next_value]
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * self.masks[step] - values[step]
            gae = delta + self.gamma * self.tau * self.masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns
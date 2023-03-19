import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
LEARNING_RATE = 1e-3
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
EPISODES = 500

env = gym.make("Acrobot-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

class DQNAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNAgent, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

policy_net = DQNAgent(state_dim, action_dim)
target_net = DQNAgent(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

memory = ReplayMemory(MEMORY_SIZE)
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

def select_action(state, epsilon):
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(env.action_space.n)]], dtype=torch.long)

def optimize_model(epsilon):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    state_batch, action_batch, reward_batch, next_state_batch = zip(*transitions)

    state_batch = torch.cat(state_batch)
    action_batch = torch.cat(action_batch)
    reward_batch = torch.cat(reward_batch)
    next_state_batch = torch.cat(next_state_batch)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def train(env):
    episode_rewards = []
    epsilon = EPS_START
    for episode in range(EPISODES):
        state = env.reset()
        state = torch.tensor([state], dtype=torch.float32)
        total_reward = 0

        while True:
            action = select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.tensor([next_state], dtype=torch.float32)

            if done:
                reward = -1
            else:
                reward = 0

            reward = torch.tensor([reward], dtype=torch.float32)

            memory.push((state, action, reward, next_state))

            state = next_state
            total_reward += reward.item()

            optimize_model(epsilon)

            if done:
                episode_rewards.append(total_reward)
                break

        if (episode + 1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(EPS_END, EPS_DECAY * epsilon)  # Decay epsilon

    return episode_rewards

def plot_rewards(rewards, title):
    plt.plot(rewards, label="DQN Agent")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # Train the DQN agent
    episode_rewards = train(env)

    # Plot the rewards
    plot_rewards(episode_rewards, "Acrobot-v1 - DQN Agent vs. Random Agent")

    
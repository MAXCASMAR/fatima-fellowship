import os
os.system('pip install gym')

import random
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch import nn
from collections import deque,namedtuple

os.system('apt update')
os.system('apt-get install python-opengl -y')
os.system('apt install xvfb -y')
os.system('pip install pyvirtualdisplay')
os.system('pip install piglet')

import glob
import io
import base64
import os
from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
#from gym.wrappers import Monitor

# Implement Deep Q network
class DQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super().__init__()
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim

        self.fc1 = nn.Linear(self.state_space_dim, 128)
        self.fc2 = nn.Linear(64, 128)
        self.head = nn.Linear(128, self.action_space_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.head(x)

# Experience replay
# NNs may be unstable or diverge when exists correlation in the observations. 
# Experience replay allows agents to remember and learn from past experiences
# It randomizes data, so we have uncorrelated data
# Store experience in t e_t (state, action, next_state, reward) for every t, in each step, in a dataset D = {e_1, e_2, ..., e_t}
# During learning, apply Q-learning updates on samples (or batches) of experiences drawn uniformly at random from D, (s, a, r, s')~U(D)
# Use a queue with a fixed capacity. When it is reached, the oldest element is replaced with the new one, use deque library

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        # Add the tuple (state, action, next_state, reward) to the queue
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self)) 
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory) 

# Epsilon-greedy policy 
# After the experience replay, then select and perform an action according to this policy.
# Choose a rnadom action with probability epsilon, otherwise choose the best action corresponding to the 
# highest Q-value. This is done so that the agent explores the env rather than only doing explotation. 
# With exploration, the agent can learn more about the env and find better policies.
def choose_action_epsilon_greedy(net, state, epsilon):
    
    if epsilon > 1 or epsilon < 0:
        raise Exception('The epsilon value must be between 0 and 1')
                
    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32) # Convert the state to tensor
        net_out = net(state)

    # Get the best action (argmax of the network output)
    best_action = int(net_out.argmax())
    # Get the number of possible actions
    action_space_dim = net_out.shape[-1]

    # Select a non optimal action with probability epsilon, otherwise choose the best action
    if random.random() < epsilon:
        # List of non-optimal actions (this list includes all the actions but the optimal one)
        non_optimal_actions = [a for a in range(action_space_dim) if a != best_action]
        # Select randomly from non_optimal_actions
        action = random.choice(non_optimal_actions)
    else:
        # Select best action
        action = best_action
        
    return action, net_out.cpu().numpy()
    
if __name__ == '__main__':
    # action and observation spaces
    env = gym.make('Acrobot-v1')
    
    action_space = env.action_space # Discrete(3) [0, 1, 2]
    observation_space = env.observation_space # Box([ -1.        -1.        -1.        -1.       -12.566371 -28.274334], [ 1.        1.        1.        1.       12.566371 28.274334], (6,), float32)
    # each observation is an array of 6 numbers (float32)
    print('Action space:', action_space)
    print('Observation space:', observation_space)


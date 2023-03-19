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
# Store experience in t e_t

if __name__ == '__main__':
    # action and observation spaces
    env = gym.make('Acrobot-v1')

    action_space = env.action_space # Discrete(3) [0, 1, 2]
    observation_space = env.observation_space # Box([ -1.        -1.        -1.        -1.       -12.566371 -28.274334], [ 1.        1.        1.        1.       12.566371 28.274334], (6,), float32)
    # each observation is an array of 6 numbers (float32)
    print('Action space:', action_space)
    print('Observation space:', observation_space)


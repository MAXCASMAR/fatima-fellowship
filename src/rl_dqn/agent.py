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

#os.system('apt update')
#os.system('apt-get install python-opengl -y')
#os.system('apt install xvfb -y')
#os.system('pip install pyvirtualdisplay')
#os.system('pip install piglet')

import glob
import io
import base64
import os
from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
#from gym.wrappers import Monitor


#display = Display(visible=0, size=(1400, 900))
#display.start()

#if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY"))==0:
#    os.system('!bash ../xvfb start')
#    os.system('%env DISPLAY=:1')
      
def show_videos():
  mp4list = glob.glob('video/*.mp4')
  mp4list.sort()
  for mp4 in mp4list:
    print(f"\nSHOWING VIDEO {mp4}")
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    
def wrap_env(env, video_callable=None):
  env = Monitor(env, './video', force=True, video_callable=video_callable)
  return env   

# Implement Deep Q network
class DQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super().__init__()
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim

        self.fc1 = nn.Linear(self.state_space_dim, 128)
        self.fc2 = nn.Linear(128, 128)
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

# Softmax policy
# Choose the optimal action, based on a distribution obtained by applying a softmax to the 
# estimated Q-values, given the temperature. Two cases: 1) the higher the temperature, the more
# the distribution will converte to a random uniform distribution
# 2) at zero temperature, the policy will always choose the action with the highest Q-value


def choose_action_softmax(net, state, temperature):
    
    if temperature < 0:
        raise Exception('The temperature value must be greater than or equal to 0 ')
        
    # If the temperature is 0, just select the best action using the eps-greedy policy with epsilon = 0
    if temperature == 0:
        return choose_action_epsilon_greedy(net, state, 0)
    
    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32)
        net_out = net(state)

    # Apply softmax with temp
    temperature = max(temperature, 1e-8) # set a minimum to the temperature for numerical stability
    softmax_out = nn.functional.softmax(net_out/temperature, dim=0).cpu().numpy()
                
    # Sample the action using softmax output as mass pdf
    all_possible_actions = np.arange(0, softmax_out.shape[-1])
    # this samples a random element from "all_possible_actions" with the probability distribution p (softmax_out in this case)
    action = np.random.choice(all_possible_actions,p=softmax_out)
    
    return action, net_out.cpu().numpy()


# Exploration profile
# An exponentially decreasing exploration profile using the softmax policy
# softmax_temperature = initial_temperature * exponential_decay^i


def update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size):
        
    # Sample from the replay memory
    batch = replay_mem.sample(batch_size)
    batch_size = len(batch)

    # Create tensors for each element of the batch
    states      = torch.tensor([s[0] for s in batch], dtype=torch.float32, device=device)
    actions     = torch.tensor([s[1] for s in batch], dtype=torch.int64, device=device)
    rewards     = torch.tensor([s[3] for s in batch], dtype=torch.float32, device=device)

    # Compute a mask of non-final states (all the elements where the next state is not None)
    non_final_next_states = torch.tensor([s[2] for s in batch if s[2] is not None], dtype=torch.float32, device=device) # the next state can be None if the game has ended
    non_final_mask = torch.tensor([s[2] is not None for s in batch], dtype=torch.bool)

    # Compute Q values 
    policy_net.train()
    q_values = policy_net(states)
    # Select the proper Q value for the corresponding action taken Q(s_t, a)
    state_action_values = q_values.gather(1, actions.unsqueeze(1))

    # Compute the value function of the next states using the target network V(s_{t+1}) = max_a( Q_target(s_{t+1}, a)) )
    with torch.no_grad():
      target_net.eval()
      q_values_target = target_net(non_final_next_states)
    next_state_max_q_values = torch.zeros(batch_size, device=device)
    next_state_max_q_values[non_final_mask] = q_values_target.max(dim=1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = rewards + (next_state_max_q_values * gamma)
    expected_state_action_values = expected_state_action_values.unsqueeze(1)# Set the required tensor shape

    # Compute the Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Apply gradient clipping 
    nn.utils.clip_grad_norm_(policy_net.parameters(), 2)
    optimizer.step()


if __name__ == '__main__':

    ### Define exploration profile
    initial_value = 5
    num_iterations = 800
    exp_decay = np.exp(-np.log(initial_value) / num_iterations * 6) 
    exploration_profile = [initial_value * (exp_decay ** i) for i in range(num_iterations)]

        # environment
    env = gym.make('Acrobot-v1') 
    #env.seed(0) # Set a random seed for the environment 

    state_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"STATE SPACE SIZE: {state_space_dim}")
    print(f"ACTION SPACE SIZE: {action_space_dim}")

    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    gamma = 0.99  
    replay_memory_capacity = 10000   
    lr = 1e-3
    target_net_update_steps = 10   
    batch_size = 256   
    bad_state_penalty = 0   
    min_samples_for_training = 1000   

    # replay memory
    replay_mem = ReplayMemory(replay_memory_capacity)    

    # policy network
    policy_net = DQN(state_space_dim, action_space_dim).to(device)

    # target network with the same weights of the policy network
    target_net = DQN(state_space_dim, action_space_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr) # The optimizer will update ONLY the parameters of the policy network

    loss_fn = nn.SmoothL1Loss()

    env = gym.make('Acrobot-v1') 
    #env.seed(0) 

    #env = wrap_env(env, video_callable=lambda episode_id: episode_id % 100 == 0) # Save a video every 100 episodes

    plotting_rewards=[]

    for episode_num, tau in enumerate(tqdm(exploration_profile)):

        state = env.reset()
        score = 0
        done = False

        while not done:

            # Choose the action following the policy
            action, q_values = choose_action_softmax(policy_net, state, temperature=tau)
            
            next_state, reward, done, info = env.step(action)

            # Update the final score (-1 for each step)
            score += reward

            if done:  
                reward += bad_state_penalty
                next_state = None
            
            # Update the replay memory
            replay_mem.push(state, action, next_state, reward)

            # Update the network
            if len(replay_mem) > min_samples_for_training: # we enable the training only if we have enough samples in the replay memory, otherwise the training will use the same samples too often
                update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size)

            # Visually render the environment 
            #env.render()

            # Set the current state for the next iteration
            state = next_state

        # Update the target network every target_net_update_steps episodes
        if episode_num % target_net_update_steps == 0:
            print('Updating target network...')
            target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network
        
        plotting_rewards.append(score)
        print(f"EPISODE: {episode_num + 1} - FINAL SCORE: {score} - Temperature: {tau}") # Print the final score

    env.close()
    #show_videos()

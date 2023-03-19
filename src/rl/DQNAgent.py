import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from torch.autograd import Variable
from utils import *
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def deep_Q_learning(env, optimizer_spec, exploration_params, replay_buffer_size=100000,
                    start_learning=50000, batch_size=128, gamma=0.99, target_update_freq=10000,
                    save_fig=True, save_model=False):
    def select_epsilon_greedy_action(model, state, exploration_params, t):
        fraction = min(1.0, float(t) / exploration_params["timesteps"])
        epsilon = 1 + fraction * (exploration_params["final_eps"] - 1)
        num_actions = model.head.out_features

        if random.random() <= epsilon:
            return random.randrange(num_actions), epsilon
        else:
            return int(model(Variable(state)).data.argmax()), epsilon

    num_actions = env.action_space.n
    Q = DQN(num_actions).to(device)
    Q_target = DQN(num_actions).to(device)
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)
    replay_buffer = PriorReplayMemory(replay_buffer_size)

    episodic_rewards = deque(maxlen=100)
    best_avg_episodic_reward = -np.inf
    acc_episodic_reward = 0.0

    num_param_updates = 0
    episodes_passed = 0
    stopping_counter = 0

    env.reset()
    current_screen = get_screen(env)
    state = current_screen

    for t in count():
        if np.mean(episodic_rewards) > -115 and len(episodic_rewards) >= 100:
            stopping_counter += 1
            if stopping_counter >= 11:
                if save_model:
                    torch.save(Q, 'stable_trained_Acrobot_model_v4')
                break
        else:
            stopping_counter = 0

        if t > start_learning:
            action, eps_val = select_epsilon_greedy_action(Q, state, exploration_params, t)
        else:
            action = random.randrange(num_actions)
            eps_val = 1.0

        _, reward, done, _ = env.step(action)
        last_screen = current_screen
        current_screen = get_screen(env)
        next_state = current_screen - last_screen

        current_Q_value = Q(state)[0][action]
        next_Q_value = Q_target(next_state).detach().max(1)[0]
        target_Q_value = reward + (gamma * next_Q_value)
        bellman_error = target_Q_value - current_Q_value.squeeze()

        acc_episodic_reward += reward
        transition = Transition(state=state, action=action, reward=reward, next_state=next_state, done=int(done))
        replay_buffer.insert(transition, np.abs(bellman_error.data))

        if done:
            env.reset()
            current_screen = get_screen(env)
            next_state = current_screen

            episodic_rewards.append(acc_episodic_reward)
            acc_episodic_reward = 0.0
            episodes_passed += 1

            if len(episodic_rewards) > 100:
                episodic_rewards.pop(0)

            avg_episodic_rewards.append(np.mean(episodic_rewards))
            stdev_episodic_rewards.append(np.std(episodic_rewards))

            if np.mean(episodic_rewards) > best_avg_episodic_reward:
                best_avg_episodic_reward = np.mean(episodic_rewards)
                if save_model:
                    torch.save(Q, 'trained_DQN_model')

            if episodes_passed % 20 == 0:
                plot_rewards(np.array(episodic_rewards), np.array(avg_episodic_rewards),
                             np.array(stdev_episodic_rewards), save_fig)
                print('Episode {}\tAvg. Reward: {:.2f}\tEpsilon: {:.4f}\t'.format(
                    episodes_passed, avg_episodic_rewards[-1], eps_val))
                print('Best avg. episodic reward:', best_avg_episodic_reward)

        state = next_state

        if t > start_learning and replay_buffer.can_sample(batch_size):
            state_batch, action_batch, reward_batch, next_state_batch, done_mask, idxs_batch, is_weight = \
                replay_buffer.sample(batch_size)

            state_batch = torch.cat(state_batch)
            action_batch = Variable(torch.tensor(action_batch).long()).to(device)
            reward_batch = Variable(torch.tensor(reward_batch, device=device)).type(dtype)
            next_state_batch = torch.cat(next_state_batch)
            not_done_mask = Variable(1 - torch.tensor(done_mask)).type(dtype).to(device)
            is_weight = Variable(torch.tensor(is_weight)).type(dtype).to(device)

            current_Q_values = Q(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
            Q_max_next_state = Q_target(next_state_batch).detach().max(1)[0]
            next_Q_values = not_done_mask * Q_max_next_state
            target_Q_values = reward_batch + (gamma * next_Q_values)

            loss = (current_Q_values - target_Q_values.detach()).pow(2) * is_weight
            prios = loss + 1e-5
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()

            for i in range(batch_size):
                idx = idxs_batch[i]
                replay_buffer.update(idx, prios[i].data.cpu().numpy())

            optimizer.step()
            num_param_updates += 1

            if num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())



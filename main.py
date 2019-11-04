import numpy as np
import tensorflow as tf
import gym
import csv
from gym.wrappers import AtariPreprocessing

from envs import ATARI_ENVS
from agents.ddpg import DDPG
from agents.dqn import DQN
from agents.double_dqn import DoubleDQN
from agents.dueling_dqn import DuelingDQN


def main():
    discrete_agents = [DuelingDQN, DQN, DuelingDQN, DoubleDQN]
    discrete_envs = ['Breakout-v0']#ATARI_ENVS
    continuous_agents = [DDPG]
    continuous_envs = ['CarRacing-v0', 'BipedalWalker-v2', 'Pendulum-v0',
                       'LunarLanderContinuous-v2', 'BipedalWalkerHardcore-v2',
                       'MountainCarContinuous-v0',
                      ]

    evaluate_envs(discrete_envs, discrete_agents)
    evaluate_envs(continuous_envs, continuous_agents)

def evaluate_envs(envs, agents):
    for env_name in envs:
        for agent_class in agents:
            result = run_env(env_name, agent_class)
            with open('results/{}_{}.csv'.format(agent_class.__name__, env_name), 'a+') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerow(result)

def run_env(env_name, agent_class):
    """Runs an agent in a single environment to evaluate its performance."""
    env = gym.make(env_name)
    if env_name in ATARI_ENVS:
        env = AtariPreprocessing(env)
    if isinstance(env.action_space, gym.spaces.Discrete):
        num_state_feats = env.observation_space.shape
        num_actions = env.action_space.n
        agent = agent_class(env_name, num_state_feats, num_actions)
    else:
        num_state_feats = env.observation_space.shape
        num_action_feats = env.action_space.shape[0]
        min_action_values = env.action_space.low
        max_action_values = env.action_space.high
        agent = agent_class(env_name, num_state_feats, num_action_feats, min_action_values, max_action_values)

    min_observation_values = env.observation_space.low
    max_observation_values = env.observation_space.high
    normalize_state = False
    if env_name in ['CarRacing-v0', 'Pendulum-v0'] + ATARI_ENVS:
        normalize_state = True
    
    noise = 0.1
    num_episodes = 100000
    num_train_steps = 50
    batch_size = 32
    discount = 0.99
    last_100_ep_ret = []
    results = []

    for episode in range(num_episodes+1):
        # Reset environment.
        state = env.reset()
        if normalize_state:
            state = np.divide(state, max_observation_values, dtype=np.float32)
        done, ep_rew = False, 0
        while not done:
            state_in = np.expand_dims(state, axis=0)
            # Sample action from policy and take that action in the env.
            action = agent.take_exploration_action(state_in, env, noise)
            next_state, reward, done, info = env.step(action)
            if normalize_state:
                next_state = np.divide(next_state, max_observation_values, dtype=np.float32)
            ep_rew += reward
            agent.buffer.add_to_buffer(state, action, reward, next_state, done)
            state = next_state
            
        if len(last_100_ep_ret) == 100:
            last_100_ep_ret = last_100_ep_ret[1:]
        last_100_ep_ret.append(ep_rew)
        
        if episode == 0:
            print_env_info(env, env_name, agent)

        # Keep collecting experience with the current policy.
        if len(agent.buffer) < batch_size:
            continue

        # Perform training on data sampled from the replay buffer.
        for _ in range(num_train_steps):
            states, actions, rewards, next_states, dones = agent.buffer.sample(batch_size)
            states = np.reshape(states, (batch_size,) + num_state_feats).astype(np.float32)
            if isinstance(env.action_space, gym.spaces.Discrete):
                actions = np.reshape(actions, (batch_size, )).astype(np.uint8)
            else:
                actions = np.reshape(actions, (batch_size, -1)).astype(np.float32)
            rewards = np.reshape(rewards, (batch_size, )).astype(np.float32)
            next_states = np.reshape(next_states, (batch_size, ) + num_state_feats).astype(np.float32)
            dones = np.reshape(dones, (batch_size, )).astype(np.float32)
            loss_tuple = agent.train_step(states, actions, rewards, next_states, dones)

        # Print the performance of the policy.
        if episode % 100 == 0:
            if len(loss_tuple) == 1:
                loss = loss_tuple[0]
                loss_info = 'Loss: {:.2f}, '.format(loss)
            else:
                actor_loss, critic_loss = loss_tuple
                loss_info = 'Actor loss: {:.2f}, '.format(actor_loss) + \
                            'Critic loss: {:.2f} '.format(critic_loss)
            print('Episode: {}/{}, '.format(episode, num_episodes) + loss_info + \
                  'Last 100 episode return: {:.2f}'.format(np.mean(last_100_ep_ret)))

        results.append(np.mean(last_100_ep_ret))
    return results

def print_env_info(env, env_name, agent):
    print('\n\n\n=============================')
    print('Environemnt: {}'.format(env_name))
    print('Agent: {}'.format(type(agent).__name__))
    print('Observation shape: {}'.format(env.observation_space.shape))
    print('Action shape: {}'.format(env.action_space))
    print('=============================\n')

if __name__ == '__main__':
    main()

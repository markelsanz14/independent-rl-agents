import numpy as np
import tensorflow as tf
import gym
import time

from agents.ddpg import DDPG
from agents.dqn import DQN


def evaluate_discrete_envs():
    discrete_envs = ['CartPole-v0']
    for env_name in discrete_envs:
        env = gym.make(env_name)
        num_state_feats = env.observation_space.shape
        num_actions = env.action_space.n
        agent = DQN(env_name, num_state_feats, num_actions)
        epsilon = 0.2

        num_episodes = 1500
        num_train_steps = 10
        batch_size = 32
        discount = 0.99
        last_100_ep_ret = []

        for episode in range(num_episodes):
            # Reset environment.
            state = env.reset()
            done, ep_rew = False, 0
            while not done:
                state_in = np.expand_dims(state, axis=0)
                # Sample action from policy and take that action in the env.
                action = agent.take_epsilon_greedy_action(state_in, epsilon=epsilon)
                next_state, reward, done, info = env.step(action.numpy())
                #next_state = np.reshape(next_state, (1, -1))
                ep_rew += reward
                agent.buffer.add_to_buffer(state, action, reward, next_state, done)
                state = next_state

            if len(last_100_ep_ret) == 100:
                last_100_ep_ret = last_100_ep_ret[1:]
            last_100_ep_ret.append(ep_rew)

            # Keep collecting experience with the current policy.
            if len(agent.buffer) < batch_size:
                continue

            # Perform training on data sampled from the replay buffer.
            start = time.time()
            for _ in range(num_train_steps):
                states, actions, rewards, next_states, dones = agent.buffer.sample(batch_size)
                states = np.reshape(states, (batch_size,) + num_state_feats)
                actions = np.reshape(actions, (batch_size,))
                rewards = np.reshape(rewards, (batch_size,))
                next_states = np.reshape(next_states, (batch_size,) + num_state_feats)
                dones = np.reshape(dones, (batch_size,))
                loss = agent.train_step(states.astype(np.float32),
                                        actions.astype(np.uint8),
                                        rewards.astype(np.float32),
                                        next_states.astype(np.float32),
                                        dones.astype(np.float32))
            end = time.time()

            if episode == 0:
                print('\n\n\n=============================')
                print('Environemnt: {}'.format(env_name))
                print('Observation shape: {}'.format(env.observation_space.shape))
                print('Number of Actions: {}'.format(env.action_space.n))
                print('=============================\n')

            # Print the performance of the policy.
            if episode % 50 == 0:
                #ipythondisplay.clear_output()
                print('Episode: {}/{}, '.format(episode, num_episodes) + \
                      'Loss: {:.2f}, '.format(loss) + \
                      'Last 100 episode return: {:.2f}'.format(np.mean(last_100_ep_ret)))
                #print(text)
                #print('Current agent performance:')
                #show_video()


def main():
    evaluate_discrete_envs()

def evaluate_continuous_envs():
    continuous_envs = ['BipedalWalker-v2', 'CarRacing-v0', 'Pendulum-v0', 'LunarLanderContinuous-v2', 'BipedalWalkerHardcore-v2', 'MountainCarContinuous-v0']
    for env_name in continuous_envs:
        env = gym.make(env_name)
        num_state_feats = env.observation_space.shape
        num_action_feats = env.action_space.shape[0]
        min_action_values = env.action_space.low
        max_action_values = env.action_space.high
        agent = DDPG(env_name, num_state_feats, num_action_feats, min_action_values, max_action_values)

        min_observation_values = env.observation_space.low
        max_observation_values = env.observation_space.high
        normalize_state = False
        if env_name in ['CarRacing-v0', 'Pendulum']:
            normalize_state = True
        noise = 0.1
        if env_name == 'MountainCarContinuous-v0':
            noise = 0.5

        num_episodes = 3000
        num_train_steps = 500
        batch_size = 64
        discount = 0.99
        last_100_ep_ret = []

        for episode in range(num_episodes):
            # Reset environment.
            state = env.reset()
            done, ep_rew = False, 0
            while not done:
                state_in = np.expand_dims(state, axis=0)
                # Sample action from policy and take that action in the env.
                action = agent.take_action_with_noise(state_in, noise_scale=noise)
                next_state, reward, done, info = env.step(action[0].numpy())
                #next_state = np.reshape(next_state, (1, -1))
                if normalize_state:
                    next_state = np.divide(next_state, max_observation_values, dtype=np.float32)
                ep_rew += reward
                agent.buffer.add_to_buffer(state, action[0], reward, next_state, done)
                state = next_state

            if len(last_100_ep_ret) == 100:
                last_100_ep_ret = last_100_ep_ret[1:]
            last_100_ep_ret.append(ep_rew)

            # Keep collecting experience with the current policy.
            if len(agent.buffer) < batch_size:
                continue

            # Perform training on data sampled from the replay buffer.
            start = time.time()
            for _ in range(num_train_steps):
                states, actions, rewards, next_states, dones = agent.buffer.sample(batch_size)
                states = np.reshape(states, (batch_size,) + num_state_feats)
                actions = np.reshape(actions, (batch_size, -1))
                rewards = np.reshape(rewards, (batch_size,))
                next_states = np.reshape(next_states, (batch_size,) + num_state_feats)
                dones = np.reshape(dones, (batch_size,))
                critic_loss, actor_loss = agent.train_step(states.astype(np.float32),
                                                           actions.astype(np.float32),
                                                           rewards.astype(np.float32),
                                                           next_states.astype(np.float32),
                                                           dones.astype(np.float32))
            end = time.time()

            if episode == 0:
                print('\n\n\n=============================')
                print('Environemnt: {}'.format(env_name))
                print('Observation shape: {}'.format(env.observation_space.shape))
                print('Action shape: {}'.format(env.action_space.shape))
                print('Min action values: {}'.format(min_action_values))
                print('Max action values: {}'.format(max_action_values))
                if env_name == 'CarRacing-v0':
                    print('Min observation values: {}'.format(min_observation_values[0, 0, 0]))
                    print('Max observation values: {}'.format(max_observation_values[0, 0, 0]))
                else:
                    print('Min observation values: {}'.format(min_observation_values))
                    print('Max observation values: {}'.format(max_observation_values))
                print('=============================\n')

            # Print the performance of the policy.
            if episode % 20 == 0:
                #ipythondisplay.clear_output()
                print('Episode: {}/{}, '.format(episode, num_episodes) + \
                      'Critic Loss: {:.2f}, '.format(critic_loss) + \
                      'Actor Loss: {:.2f}, '.format(actor_loss) + \
                      'Last 100 episode return: {:.2f}'.format(np.mean(last_100_ep_ret)))
                #print(text)
                #print('Current agent performance:')
                #show_video()

if __name__ == '__main__':
    main()

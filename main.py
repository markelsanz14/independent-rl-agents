import numpy as np
import tensorflow as tf
import gym
import time

from agents.ddpg import DDPG


def main():
    envs = ['Pendulum-v0', 'LunarLanderContinuous-v2', 'BipedalWalker-v2', 'BipedalWalkerHardcore-v2',]
    for env_name in envs:
        env = gym.make(env_name)
        num_state_feats = env.observation_space.shape[0]
        num_action_feats = env.action_space.shape[0]
        min_action_values = env.action_space.low
        max_action_values = env.action_space.high
        agent = DDPG(num_state_feats, num_action_feats, min_action_values, max_action_values)

        min_observation_values = env.observation_space.low
        max_observation_values = env.observation_space.high
        normalize_state = False

        num_episodes = 2000
        num_train_steps = 100
        batch_size = 3
        discount = 0.99
        last_100_ep_ret = []

        for episode in range(num_episodes):
            # Reset environment.
            state = env.reset()
            if normalize_state:
                state /= max_observation_values
            done, ep_rew = False, 0

            while not done:
                state_in = np.reshape(state, (1, -1))
                # Sample action from policy and take that action in the env.
                action = agent.take_action_with_noise(state_in, noise_scale=0.1)
                next_state, reward, done, info = env.step(action[0])
                next_state = np.reshape(next_state, (1, -1))
                if normalize_state:
                    next_state /= max_observation_values
                ep_rew += reward
                agent.buffer.add_to_buffer(state_in[0], action[0], reward, next_state[0], done)
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
                states = np.reshape(states, (batch_size, -1))
                actions = np.reshape(actions, (batch_size, -1))
                rewards = np.reshape(rewards, (batch_size,))
                next_states = np.reshape(next_states, (batch_size, -1))
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
                print('Min observation values: {}'.format(min_observation_values))
                print('Max observation values: {}'.format(max_observation_values))
                print('=============================\n')

            # Print the performance of the policy.
            if episode % 50 == 0:
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

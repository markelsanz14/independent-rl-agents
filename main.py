import numpy as np
import tensorflow as tf
import gym

from agents.ddpg import DDPG
from agents.dqn import DQN



def main():
    discrete_action_agents = ['DQN', 'DoubleDQN', 'DuelingDQN']
    continuous_action_agents = ['SimplePolicyGradient', 'DDPG']
    discrete_envs = {'CartPole-v0': [DQN]}#, DoubleDQN, DuelingDQN]}
    continuous_envs = {'CarRacing-v0': [DDPG],
                       'BipedalWalker-v2': [DDPG],
                       'Pendulum-v0': [DDPG],
                       'LunarLanderContinuous-v2': [DDPG],
                       'BipedalWalkerHardcore-v2': [DDPG],
                       'MountainCarContinuous-v0': [DDPG],}

    evaluate_continuous_envs(continuous_envs)
    evaluate_discrete_envs(discrete_envs)

def evaluate_discrete_envs(discrete_envs):
    for env_name, agent_list in discrete_envs.items():
        for agent_class in agent_list:
            run_env(env_name, agent_class)

def evaluate_continuous_envs(continuous_envs):
    for env_name, agent_list in continuous_envs.items():
        for agent_class in agent_list:
            run_env(env_name, agent_class)

def run_env(env_name, agent_class):
    """Runs an agent in a single environment to evaluate its performance."""
    env = gym.make(env_name)
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
    if env_name in ['CarRacing-v0', 'Pendulum']:
        normalize_state = True
    
    noise = 0.1
    num_episodes = 1000
    num_train_steps = 500
    batch_size = 64
    discount = 0.99
    last_100_ep_ret = []

    for episode in range(num_episodes+1):
        # Reset environment.
        state = env.reset()
        done, ep_rew = False, 0
        while not done:
            state_in = np.expand_dims(state, axis=0)
            # Sample action from policy and take that action in the env.
            action = agent.take_exploration_action(state_in, noise)
            next_state, reward, done, info = env.step(action.numpy())
            if normalize_state:
                next_state = np.divide(next_state, max_observation_values, dtype=np.float32)
            ep_rew += reward
            agent.buffer.add_to_buffer(state, action, reward, next_state, done)
            state = next_state

        if len(last_100_ep_ret) == 100:
            last_100_ep_ret = last_100_ep_ret[1:]
        last_100_ep_ret.append(ep_rew)
        
        if episode == 0:
            print_env_info(env, env_name)

        # Keep collecting experience with the current policy.
        if len(agent.buffer) < batch_size:
            continue

        # Perform training on data sampled from the replay buffer.
        for _ in range(num_train_steps):
            states, actions, rewards, next_states, dones = agent.buffer.sample(batch_size)
            states = np.reshape(states, (batch_size,) + num_state_feats)
            if isinstance(env.action_space, gym.spaces.Discrete):
                actions = np.reshape(actions, (batch_size, )).astype(np.uint8)
            else:
                actions = np.reshape(actions, (batch_size, -1)).astype(np.float32)
            rewards = np.reshape(rewards, (batch_size, ))
            next_states = np.reshape(next_states, (batch_size, ) + num_state_feats)
            dones = np.reshape(dones, (batch_size, ))
            loss_tuple = agent.train_step(states.astype(np.float32),
                                          actions,
                                          rewards.astype(np.float32),
                                          next_states.astype(np.float32),
                                          dones.astype(np.float32))

        # Print the performance of the policy.
        if episode % 20 == 0:
            if len(loss_tuple) == 1:
                loss = loss_tuple[0]
                loss_info = 'Loss: {:.2f}, '.format(loss)
            else:
                actor_loss, critic_loss = loss_tuple
                loss_info = 'Actor loss: {:.2f}, '.format(actor_loss) + \
                            'Critic loss: {:.2f} '.format(critic_loss)
            print('Episode: {}/{}, '.format(episode, num_episodes) + loss_info + \
                  'Last 100 episode return: {:.2f}'.format(np.mean(last_100_ep_ret)))

def print_env_info(env, env_name):
    print('\n\n\n=============================')
    print('Environemnt: {}'.format(env_name))
    print('Observation shape: {}'.format(env.observation_space.shape))
    print('Action shape: {}'.format(env.action_space))
    print('=============================\n')

if __name__ == '__main__':
    main()

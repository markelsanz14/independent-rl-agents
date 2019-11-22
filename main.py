import csv

import numpy as np
import tensorflow as tf
import gym
from gym.wrappers import AtariPreprocessing

from envs import ATARI_ENVS
from agents.ddpg import DDPG
from agents.dqn import DQN
from agents.double_dqn import DoubleDQN
from agents.double_dueling_dqn import DoubleDuelingDQN
from agents.dueling_dqn import DuelingDQN


def main():
    """Main function. It runs the different algorithms in all the environemnts.
    """
    discrete_agents = [DoubleDuelingDQN, DQN, DuelingDQN, DoubleDQN]
    discrete_envs = ATARI_ENVS
    continuous_agents = [DDPG]
    continuous_envs = [
        "CarRacing-v0",
        "BipedalWalker-v2",
        "Pendulum-v0",
        "LunarLanderContinuous-v2",
        "BipedalWalkerHardcore-v2",
        "MountainCarContinuous-v0",
    ]
    evaluate_envs(discrete_envs, discrete_agents)
    evaluate_envs(continuous_envs, continuous_agents)


def evaluate_envs(envs, agents):
    """Runs all the environments with all the agents paseed in the lists.
    Args:
        envs: list, a list of gym environemnt names.
        agents: list, a list of agent classes.
    """
    clip_rewards = True
    for agent_class in agents:
        for env_name in envs:
            result = run_env(env_name, agent_class, clip_rewards)
            with open(
                "results/{}_{}_ClipRew{}.csv".format(
                    agent_class.__name__, env_name, clip_rewards
                ),
                "a+",
            ) as writeFile:
                writer = csv.writer(writeFile)
                writer.writerow(result)


def run_env(env_name, agent_class, clip_rewards=True):
    """Runs an agent in a single environment to evaluate its performance.
    Args:
        env_name: str, name of a gym environment.
        agent_class: class object, one of the agents in the agent directory.
        clip_rewards: bool, whether to clip the rewards to {-1, 0, 1} or not.
    Returns:
        result: list, the list of returns for each episode.
    """
    env = gym.make(env_name)
    if env_name in ATARI_ENVS:
        env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
    if isinstance(env.action_space, gym.spaces.Discrete):
        num_state_feats = env.observation_space.shape
        num_actions = env.action_space.n
        agent = agent_class(env_name, num_state_feats, num_actions)
    else:
        num_state_feats = env.observation_space.shape
        num_action_feats = env.action_space.shape[0]
        min_action_values = env.action_space.low
        max_action_values = env.action_space.high
        agent = agent_class(
            env_name,
            num_state_feats,
            num_action_feats,
            min_action_values,
            max_action_values,
        )

    noise = 0.1
    num_episodes = 200000
    num_train_steps = 50
    batch_size = 32
    last_100_ep_ret = []
    results = []

    for episode in range(num_episodes + 1):
        # Reset environment.
        state = env.reset()
        done, ep_rew = False, 0
        while not done:
            state_in = tf.expand_dims(state, axis=0)
            # Sample action from policy and take that action in the env.
            action = agent.take_exploration_action(state_in, env, noise)
            next_state, reward, done, info = env.step(action)
            ep_rew += reward
            if clip_rewards:
                if reward < 0:
                    reward = -1.0
                elif reward > 0:
                    reward = 1.0
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
            states, actions, rewards, next_states, dones = agent.buffer.sample(
                batch_size
            )
            states = tf.reshape(states, (batch_size,) + num_state_feats)
            if isinstance(env.action_space, gym.spaces.Discrete):
                actions = tf.cast(tf.reshape(actions, (batch_size,)), tf.uint8)
            else:
                actions = tf.reshape(actions, (batch_size, -1))
            rewards = tf.reshape(rewards, (batch_size,))
            next_states = tf.reshape(
                next_states, (batch_size,) + num_state_feats
            )
            dones = tf.cast(tf.reshape(dones, (batch_size,)), tf.float32)
            loss_tuple = agent.train_step(
                states, actions, rewards, next_states, dones
            )

        # Print the performance of the policy.
        if episode % 100 == 0:
            if len(loss_tuple) == 1:
                loss = loss_tuple[0]
                loss_info = "Loss: {:.2f}, ".format(loss)
            else:
                actor_loss, critic_loss = loss_tuple
                loss_info = "Actor loss: {:.2f}, ".format(
                    actor_loss
                ) + "Critic loss: {:.2f} ".format(critic_loss)
            print(
                "Episode: {}/{}, ".format(episode, num_episodes)
                + loss_info
                + "Last 100 episode return: {:.2f}".format(
                    np.mean(last_100_ep_ret)
                )
            )

        results.append(np.mean(last_100_ep_ret))
    return results


def print_env_info(env, env_name, agent):
    print("\n\n\n=============================")
    print("Environemnt: {}".format(env_name))
    print("Agent: {}".format(type(agent).__name__))
    print("Observation shape: {}".format(env.observation_space.shape))
    print("Action shape: {}".format(env.action_space))
    print("=============================\n")


if __name__ == "__main__":
    main()

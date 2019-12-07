import csv
import os

import numpy as np
import tensorflow as tf
import gym
from gym.wrappers import AtariPreprocessing, FrameStack

from envs import ATARI_ENVS
from agents.ddpg import DDPG
from agents.dqn import DQN
from agents.double_dqn import DoubleDQN
from agents.double_dueling_dqn import DoubleDuelingDQN
from agents.dueling_dqn import DuelingDQN


def main():
    """Main function. It runs the different algorithms in all the environemnts.
    """
    discrete_agents = [DQN]  # , DuelingDQN, DoubleDQN, DoubleDuelingDQN]
    discrete_envs = ATARI_ENVS[0:1]
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
    # evaluate_envs(continuous_envs, continuous_agents)


def evaluate_envs(envs, agents):
    """Runs all the environments with all the agents paseed in the lists.
    Args:
        envs: list, a list of gym environemnt names.
        agents: list, a list of agent classes.
    """
    clip_rewards = True
    for agent_class in agents:
        for env_name in envs:
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            run_env(
                env_name,
                agent_class,
                prioritized=False,
                clip_rewards=clip_rewards,
            )


def run_env(env_name, agent_class, prioritized=False, clip_rewards=True):
    """Runs an agent in a single environment to evaluate its performance.
    Args:
        env_name: str, name of a gym environment.
        agent_class: class object, one of the agents in the agent directory.
        clip_rewards: bool, whether to clip the rewards to {-1, 0, 1} or not.
    Returns:
        result: list, the list of returns for each episode.
    """
    print(prioritized)
    env = gym.make(env_name)
    if env_name in ATARI_ENVS:
        env = FrameStack(
            AtariPreprocessing(env, grayscale_obs=True, scale_obs=True),
            num_stack=4,
        )

    if isinstance(env.action_space, gym.spaces.Discrete):
        num_state_feats = env.observation_space.shape
        num_actions = env.action_space.n
        agent = agent_class(
            env_name, num_state_feats, num_actions, prioritized=prioritized
        )
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

    # Creaete TensorBoard Metrics.
    log_dir = "logs/{}_{}_ClipRew{}".format(
        agent_class.__name__, env_name, clip_rewards
    )
    summary_writer = tf.summary.create_file_writer(log_dir)

    noise = 1.0
    batch_size = 32
    importances = np.array([1.0 for _ in range(batch_size)])
    beta = 0.7

    num_frames = 2000000
    cur_frame, episode = 0, 0

    while cur_frame < num_frames:
        # Reset environment.
        state = env.reset()
        done, ep_rew = False, 0
        while not done:
            state_in = tf.expand_dims(state, axis=0)
            # Sample action from policy and take that action in the env.
            action = agent.take_exploration_action(state_in, env, noise)
            next_state, reward, done, info = env.step(action)
            cur_frame += 1
            ep_rew += reward
            if clip_rewards:
                if reward < 0:
                    reward = -1.0
                elif reward > 0:
                    reward = 1.0
            agent.buffer.add_to_buffer(state, action, reward, next_state, done)
            state = next_state

            if len(agent.buffer) >= batch_size:
                # Perform training on data sampled from the replay buffer.
                if prioritized:
                    states, actions, rewards, next_states, dones, importances, indices = agent.buffer.sample(
                        batch_size
                    )
                else:
                    states, actions, rewards, next_states, dones = agent.buffer.sample(
                        batch_size
                    )
                    beta = 0.0
                loss_tuple, td_errors = agent.train_step(
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    importances ** beta,
                )
                if prioritized:
                    # Update priorities
                    agent.buffer.update_priorities(indices, td_errors)

            if cur_frame < 1e6:
                noise -= 9e-7
            elif cur_frame == 1e6:
                noise = 0.1

            if cur_frame % 10000 == 0:
                agent.target_nn.set_weights(agent.main_nn.get_weights())

            if cur_frame % 5000000 == 0 and cur_frame > 0:
                agent.save_checkpoint()

            # Add TensorBoard Summaries.
            if cur_frame % 500000:
                with summary_writer.as_default():
                    tf.summary.scalar("epsilon", noise, step=cur_frame)

        with summary_writer.as_default():
            tf.summary.scalar("return", ep_rew, step=episode)

        if episode == 0:
            print_env_info(env, env_name, agent)

        episode += 1


def print_env_info(env, env_name, agent):
    print("\n\n\n=============================")
    print("Environemnt: {}".format(env_name))
    print("Agent: {}".format(type(agent).__name__))
    print("Observation shape: {}".format(env.observation_space.shape))
    print("Action shape: {}".format(env.action_space))
    print("=============================\n")


if __name__ == "__main__":
    main()

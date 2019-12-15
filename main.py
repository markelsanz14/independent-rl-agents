import os
import argparse

import numpy as np
import tensorflow as tf
import gym

from envs import ATARI_ENVS
from atari_wrappers import make_atari, wrap_deepmind

# from agents.ddpg import DDPG
from agents.dqn import DQN
from agents.double_dqn import DoubleDQN
from agents.double_dueling_dqn import DoubleDuelingDQN
from agents.dueling_dqn import DuelingDQN


def main():
    """Main function. It runs the different algorithms in all the environemnts.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Process inputs",
    )

    parser.add_argument(
        "--env", type=str, choices=ATARI_ENVS, default="BreakoutNoFrameskip-v4"
    )
    parser.add_argument("--agent", type=str, choices=["DQN"], default="DQN")
    parser.add_argument("--prioritized", type=bool, default=False)
    parser.add_argument("--double_q", type=bool, default=False)
    parser.add_argument("--dueling", type=bool, default=False)
    parser.add_argument("--num_steps", type=int, default=int(1e7))
    parser.add_argument("--clip_rewards", type=bool, default=True)

    args = parser.parse_args()
    print("Arguments received:")
    print(args)
    if args.agent == "DQN":
        agent_class = DQN
        if args.double_q:
            if args.dueling:
                agent_class = DoubleDuelingDQN
            else:
                agent_class = DoubleDQN
        elif args.dueling:
            agent_class = DuelingDQN

    run_env(
        env_name=args.env,
        agent_class=agent_class,
        prioritized=args.prioritized,
        clip_rewards=args.clip_rewards,
        num_steps=args.num_steps,
    )

    """
    continuous_agents = [DDPG]
    continuous_envs = [
        "CarRacing-v0",
        "BipedalWalker-v2",
        "Pendulum-v0",
        "LunarLanderContinuous-v2",
        "BipedalWalkerHardcore-v2",
        "MountainCarContinuous-v0",
    ]
    """


def run_env(
    env_name,
    agent_class,
    prioritized=False,
    prioritization_alpha=0.6,
    clip_rewards=True,
    num_steps=int(1e6),
    batch_size=32,
    initial_exploration=1.0,
    learning_starts=int(1e4),
    train_freq=4,
    target_update_freq=int(1e4),
    save_ckpt_freq=int(1e6),
):
    """Runs an agent in a single environment to evaluate its performance.
    Args:
        env_name: str, name of a gym environment.
        agent_class: class object, one of the agents in the agent directory.
        prioritized: bool, whether to use prioritized experience replay.
        clip_rewards: bool, whether to clip the rewards to {-1, 0, 1} or not.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    if env_name in ATARI_ENVS:
        env = make_atari(env_name)
        env = wrap_deepmind(
            env, frame_stack=True, scale=False, clip_rewards=clip_rewards
        )
    else:
        env = gym.make(env_name)

    if isinstance(env.action_space, gym.spaces.Discrete):
        num_state_feats = env.observation_space.shape
        num_actions = env.action_space.n
        agent = agent_class(
            env_name,
            num_state_feats,
            num_actions,
            prioritized=prioritized,
            prioritization_alpha=prioritization_alpha,
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

    print_env_info(env, env_name, agent)

    # Creaete TensorBoard Metrics.
    log_dir = "logs/{}_{}_ClipRew{}".format(
        agent_class.__name__, env_name, clip_rewards
    )
    summary_writer = tf.summary.create_file_writer(log_dir)

    imp = np.array([1.0 for _ in range(batch_size)])
    beta = 0.7

    noise = initial_exploration
    returns = []
    cur_frame, episode = 0, 0

    # Start learning!
    while cur_frame < num_steps:
        state = env.reset()
        done, ep_rew = False, 0
        # Start an episode.
        while not done:
            state_in = np.array(
                np.expand_dims(state, axis=0), dtype=np.float32
            )
            # Sample action from policy and take that action in the env.
            action = agent.take_exploration_action(state_in, env, noise)
            next_state, reward, done, info = env.step(action)
            agent.buffer.add_to_buffer(state, action, reward, next_state, done)
            cur_frame += 1
            ep_rew += reward
            state = next_state

            if cur_frame > learning_starts and cur_frame % train_freq == 0:
                # Perform training on data sampled from the replay buffer.
                if prioritized:
                    st, act, rew, next_st, d, imp, indx = agent.buffer.sample(
                        batch_size
                    )
                else:
                    st, act, rew, next_st, d = agent.buffer.sample(batch_size)
                    beta = 0.0
                loss_tuple, td_errors = agent.train_step(
                    st, act, rew, next_st, d, imp ** beta
                )
                if prioritized:
                    # Update priorities
                    agent.buffer.update_priorities(indx, td_errors)

            # Update value of the exploration value epsilon.
            noise = update_noise(noise, cur_frame)

            if (
                cur_frame % target_update_freq == 0
                and cur_frame > learning_starts
            ):
                # Copy weights from main to target network.
                agent.target_nn.set_weights(agent.main_nn.get_weights())

            if cur_frame % save_ckpt_freq == 0 and cur_frame > learning_starts:
                agent.save_checkpoint()

            # Add TensorBoard Summaries.
            if cur_frame % 100000 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar("epsilon", noise, step=cur_frame)

        with summary_writer.as_default():
            tf.summary.scalar("return", ep_rew, step=episode)

        episode += 1
        returns.append(ep_rew)

        if episode % 100 == 0:
            print_result(env_name, noise, episode, cur_frame, returns)
            returns = []


def update_noise(noise, step):
    if step < 1e6:
        noise -= 9e-7
    elif step == 1e6:
        noise = 0.01
    return noise


def print_result(env_name, epsilon, episode, step, returns):
    print("-------------------------------")
    print("| Env: {:>20s}   |".format(env_name[:-14]))
    print("| Exploration time %: {:>4d}%   |".format(int(epsilon * 100)))
    print("| Episode: {:>16d}   |".format(episode))
    print("| Steps: {:>18d}   |".format(step))
    print("| Last 100 return: {:>8.2f}   |".format(np.mean(returns)))
    print("-------------------------------")


def print_env_info(env, env_name, agent):
    print("\n\n\n=============================")
    print("Environemnt: {}".format(env_name))
    print("Agent: {}".format(type(agent).__name__))
    print("Observation shape: {}".format(env.observation_space.shape))
    print("Action shape: {}".format(env.action_space))
    print("=============================\n")


if __name__ == "__main__":
    main()

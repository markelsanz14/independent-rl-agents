"""Main entry point. Used to train the agents on some env."""
import time
import argparse

import numpy as np
import tensorflow as tf
import gym

from envs import ATARI_ENVS
from atari_wrappers import make_atari, wrap_deepmind
from replay_buffers.uniform import UniformBuffer, DatasetUniformBuffer
from networks.nature_cnn import NatureCNN
from networks.dueling_cnn import DuelingCNN
from agents.dqn import DQN
from agents.double_dqn import DoubleDQN


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
    parser.add_argument("--prioritized", type=int, default=0)
    parser.add_argument("--double_q", type=int, default=0)
    parser.add_argument("--dueling", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=int(1e7))
    parser.add_argument("--clip_rewards", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=int(1e5))
    parser.add_argument("--use_dataset_buffer", type=int, default=0)

    args = parser.parse_args()
    print("Arguments received:")
    print(args)
    if args.agent == "DQN":
        agent_class = DQN
        if args.double_q:
            agent_class = DoubleDQN

    run_env(
        env_name=args.env,
        agent_class=agent_class,
        buffer_size=args.buffer_size,
        dueling=args.dueling,
        prioritized=args.prioritized,
        clip_rewards=args.clip_rewards,
        num_steps=args.num_steps,
        use_dataset_buffer=args.use_dataset_buffer,
    )


def run_env(
    env_name,
    agent_class,
    buffer_size=int(1e5),
    dueling=False,
    prioritized=False,
    prioritization_alpha=0.6,
    clip_rewards=True,
    normalize_obs=True,
    num_steps=int(1e6),
    use_dataset_buffer=False,
    batch_size=32,
    initial_exploration=1.0,
    final_exploration=0.01,
    exploration_steps=int(2e6),
    learning_starts=int(1e4),
    train_freq=1,
    target_update_freq=int(1e5),
    save_ckpt_freq=int(1e6),
):
    """Runs an agent in a single environment to evaluate its performance.
    Args:
        env_name: str, name of a gym environment.
        agent_class: class object, one of the agents in the agent directory.
        prioritized: bool, whether to use prioritized experience replay.
        clip_rewards: bool, whether to clip the rewards to {-1, 0, 1} or not.
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    limit_gpu_memory = True
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, limit_gpu_memory)

    if env_name in ATARI_ENVS:
        env = make_atari(env_name)
        env = wrap_deepmind(env, frame_stack=True, scale=False, clip_rewards=False)
    else:
        env = gym.make(env_name)

    if isinstance(env.action_space, gym.spaces.Discrete):
        num_state_feats = env.observation_space.shape
        num_actions = env.action_space.n

        if use_dataset_buffer:
            replay_buffer = DatasetUniformBuffer(size=buffer_size)
            model_input = replay_buffer.build_iterator(batch_size)
        else:
            replay_buffer = UniformBuffer(size=buffer_size)

        if dueling:
            main_network = DuelingCNN(num_actions)
            target_network = DuelingCNN(num_actions)
        else:
            main_network = NatureCNN(num_actions)
            target_network = NatureCNN(num_actions)

        agent = agent_class(
            env_name,
            num_actions=num_actions,
            main_nn=main_network,
            target_nn=target_network,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
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

    print_env_info(env, env_name, agent, main_network)

    # Create TensorBoard Metrics and save graph.
    log_dir = "logs/{}_{}_{}_ClipRews{}".format(
        agent_class.__name__, type(main_network).__name__, env_name, clip_rewards
    )
    summary_writer = tf.summary.create_file_writer(log_dir)
    profile = False
    tf.summary.trace_on(graph=True, profiler=False)
    agent.main_nn(np.random.randn(1, 84, 84, 4))
    with summary_writer.as_default():
        tf.summary.trace_export(name="trace", step=0)
    if profile:
        tf.summary.trace_on(graph=False, profiler=True)

    imp = np.array([1.0 for _ in range(batch_size)])
    beta = 0.7

    epsilon = initial_exploration
    returns, clipped_returns = [], []
    cur_frame, episode = 0, 0

    start = time.time()
    # Start learning!
    while cur_frame < num_steps:
        state = env.reset()
        done, ep_rew, clipped_ep_rew = False, 0, 0
        # Start an episode.
        while not done:
            # Sample action from policy and take that action in the env.
            state_in = np.array(state) / 255.0 if normalize_obs else state
            action = agent.take_exploration_action(state_in, env, epsilon)
            next_state, reward, done, info = env.step(action)
            clipped_reward = np.sign(reward)
            rew = clipped_reward if clip_rewards else reward
            agent.buffer.add_to_buffer(state, action, rew, next_state, done)
            cur_frame += 1
            ep_rew += reward
            clipped_ep_rew += clipped_reward
            state = next_state

            if cur_frame > learning_starts and cur_frame % train_freq == 0:
                # Perform training on data sampled from the replay buffer.
                if prioritized:
                    st, act, rew, next_st, d, imp, indx = agent.buffer.sample(
                        batch_size
                    )
                else:
                    if type(replay_buffer).__name__ == "UniformBuffer":
                        st, act, rew, next_st, d = agent.buffer.sample(batch_size)
                    else:
                        st, act, rew, next_st, d = next(model_input)
                    beta = 0.0
                    if normalize_obs:
                        st = tf.cast(st, tf.float32) / 255.0
                        next_st = tf.cast(next_st, tf.float32) / 255.0
                loss_tuple, td_errors = agent.train_step(
                    st, act, rew, next_st, d, imp ** beta
                )
                if prioritized:
                    # Update priorities
                    agent.buffer.update_priorities(indx, td_errors)

            # Update value of the exploration value epsilon.
            epsilon = decay_epsilon(
                epsilon,
                cur_frame,
                initial_exploration,
                final_exploration,
                exploration_steps,
            )

            if cur_frame % target_update_freq == 0 and cur_frame > learning_starts:
                # Copy weights from main to target network.
                agent.target_nn.set_weights(agent.main_nn.get_weights())

            if cur_frame % save_ckpt_freq == 0 and cur_frame > learning_starts:
                agent.save_checkpoint()

            # Add TensorBoard Summaries.
            if cur_frame % 100000 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar("epsilon", epsilon, step=cur_frame)

            if cur_frame == 100 and profile:
                with summary_writer.as_default():
                    tf.summary.trace_export(
                        name="trace", step=cur_frame, profiler_outdir=log_dir
                    )
            """
            if cur_frame % 100 == 0:
                end = time.time()
                print(end-start)
                start = time.time()
            """

        with summary_writer.as_default():
            tf.summary.scalar("clipped_return", clipped_ep_rew, step=episode)
            tf.summary.scalar("return", ep_rew, step=episode)

        episode += 1
        returns.append(ep_rew)
        clipped_returns.append(clipped_ep_rew)

        if episode % 100 == 0:
            print_result(
                env_name, epsilon, episode, cur_frame, returns, clipped_returns
            )
            returns = []
            clipped_returns = []


def decay_epsilon(epsilon, step, initial_exp, final_exp, exp_steps):
    if step < exp_steps:
        epsilon -= (initial_exp - final_exp) / float(exp_steps)
    return epsilon


def print_result(env_name, epsilon, episode, step, returns, clipped_returns):
    print("----------------------------------")
    print("| Env: {:>23s}   |".format(env_name[:-14]))
    print("| Exploration time %: {:>7d}%   |".format(int(epsilon * 100)))
    print("| Episode: {:>19d}   |".format(episode))
    print("| Steps: {:>21d}   |".format(step))
    print("| Last 100 return: {:>11.2f}   |".format(np.mean(returns)))
    print("| Clipped 100 return: {:>8.2f}   |".format(np.mean(clipped_returns)))
    print("----------------------------------")


def print_env_info(env, env_name, agent, network):
    print("\n\n\n=============================")
    print("Environemnt: {}".format(env_name))
    print("Agent: {}".format(type(agent).__name__))
    print("Network: {}".format(type(network).__name__))
    print("Observation shape: {}".format(env.observation_space.shape))
    print("Action shape: {}".format(env.action_space))
    print("=============================\n")


if __name__ == "__main__":
    main()

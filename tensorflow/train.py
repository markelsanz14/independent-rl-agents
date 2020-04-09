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
from networks.soft_q_net import SoftQNet
from networks.dense_sac_policy import PolicyNet
from agents.dqn import DQN
from agents.double_dqn import DoubleDQN
from agents.sac import SAC


def main():
    """Main function. It runs the different algorithms in all the environemnts.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Process inputs",
    )

    envs = ["Pendulum-v0", "LunarLanderContinuous-v2", "BipedalWalker-v2"]
    parser.add_argument(
        "--env", type=str, choices=ATARI_ENVS + envs, default="BreakoutNoFrameskip-v4"
    )
    parser.add_argument("--agent", type=str, choices=["DQN", "SAC"], default="DQN")
    parser.add_argument("--prioritized", type=int, default=0)
    parser.add_argument("--double_q", type=int, default=0)
    parser.add_argument("--dueling", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=int(1e7))
    parser.add_argument("--clip_rewards", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=int(1e5))
    parser.add_argument("--use_dataset_buffer", type=int, default=1)

    args = parser.parse_args()
    print("Arguments received:")
    print(args)
    if args.agent == "DQN":
        agent_class = DQN
        if args.double_q:
            agent_class = DoubleDQN
    elif args.agent == "SAC":
        agent_class = SAC

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


@profile
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
    learning_starts=40,  # int(1e4),
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
    print(gpus)
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
        max_state_feats = 255.0
        print(use_dataset_buffer)
        if use_dataset_buffer:
            replay_buffer = DatasetUniformBuffer(size=buffer_size, normalization_val=max_state_feats)
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
            batch_size=batch_size,
        )
        print_env_info(env, env_name, agent, main_network)
        log_dir = (
            f"logs/{agent_class.__name__}_{main_network.__name__}_"
            f"{env_name}_ClipRews{clip_rewards}"
        )
    else:
        num_state_feats = env.observation_space.shape[0]
        num_action_feats = env.action_space.shape[0]
        min_action_values = env.action_space.low
        max_action_values = env.action_space.high
        max_state_feats = env.observation_space.high
        replay_buffer = UniformBuffer(size=buffer_size)
        if agent_class.__name__ == "SAC":
            main_q1_net = SoftQNet()
            target_q1_net = SoftQNet()
            main_q2_net = SoftQNet()
            target_q2_net = SoftQNet()
            policy_net = PolicyNet(num_actions=num_action_feats)

        agent = agent_class(
            env_name=env_name,
            main_q1_nn=main_q1_net,
            main_q2_nn=main_q2_net,
            target_q1_nn=target_q1_net,
            target_q2_nn=target_q2_net,
            policy_nn=policy_net,
            target_entropy=None,
            num_action_feats=num_action_feats,
        )
        print_env_info(env, env_name, agent, policy_net)
        log_dir = f"logs/{agent_class.__name__}_{env_name}"

    # Create TensorBoard Metrics and save graph.
    summary_writer = tf.summary.create_file_writer(log_dir)
    profile = True
    with summary_writer.as_default():
        if agent.__name__ in ["DQN", "DoubleDQN"]:
            tf.summary.trace_on(graph=True, profiler=False)
            agent.main_nn(np.random.randn(1, 84, 84, 4))
            tf.summary.trace_export(name="dqn_graph", step=0)
        else:
            tf.summary.trace_on(graph=True, profiler=False)
            main_q1_net(
                np.random.randn(1, num_state_feats).astype(np.float32),
                np.random.randn(1, num_action_feats).astype(np.float32),
            )
            tf.summary.trace_export(name="q1_graph", step=0)
            # main_q2_net(
            #    np.random.randn(1, num_state_feats).astype(np.float32),
            #    np.random.randn(1, num_action_feats).astype(np.float32),
            # )
            tf.summary.trace_on(graph=True, profiler=False)
            policy_net(np.random.randn(1, num_state_feats).astype(np.float32))
            tf.summary.trace_export(name="policy_graph", step=0)
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
            state_in = np.array(state) / max_state_feats if normalize_obs else state
            action = agent.take_exploration_action(state_in, env, epsilon)
            next_state, reward, done, info = env.step(action)
            # if tf.math.is_nan(reward).numpy():
            #    quit()
            clipped_reward = np.sign(reward)
            rew = clipped_reward if clip_rewards else reward
            replay_buffer.add(state, action, rew, next_state, done)
            cur_frame += 1
            ep_rew += reward
            clipped_ep_rew += clipped_reward
            state = next_state

            if cur_frame > learning_starts and cur_frame % train_freq == 0:
                # Perform training on data sampled from the replay buffer.
                if prioritized:
                    st, act, rew, next_st, d, imp, indx = replay_buffer.sample(
                        batch_size
                    )
                else:
                    if replay_buffer.__name__ == "UniformBuffer":
                        st, act, rew, next_st, d = replay_buffer.sample(batch_size)
                    else:
                        st, act, rew, next_st, d = next(model_input)
                    beta = 0.0
                    if normalize_obs:
                        with tf.device("/gpu:0"):
                            st = tf.cast(st, tf.float32) / max_state_feats
                            next_st = tf.cast(next_st, tf.float32) / max_state_feats
                if agent.__name__ == "SAC":
                    losses_dict = agent.train_step(
                        st, act, rew, next_st, d, imp ** beta
                    )
                else:
                    loss_tuple, td_errors = agent.train_step(
                        st, act, rew, next_st, d, imp ** beta
                    )
                if cur_frame % 10 == 0 and agent.__name__ == "SAC":
                    with summary_writer.as_default():
                        for name, loss in losses_dict.items():
                            tf.summary.scalar(name, loss, step=cur_frame)

                if prioritized:
                    # Update priorities
                    replay_buffer.update_priorities(indx, td_errors)

            # Update value of the exploration value epsilon.
            epsilon = decay_epsilon(
                epsilon,
                cur_frame,
                initial_exploration,
                final_exploration,
                exploration_steps,
            )

            if (
                agent.__name__ == "DQN"
                and cur_frame % target_update_freq == 0
                and cur_frame > learning_starts
            ):
                # Copy weights from main to target network.
                agent.target_nn.set_weights(agent.main_nn.get_weights())

            if cur_frame % save_ckpt_freq == 0 and cur_frame > learning_starts:
                agent.save_checkpoint()

            # Add TensorBoard Summaries.
            if cur_frame % 100000 == 0 and agent.__name__ == "DQN":
                with summary_writer.as_default():
                    tf.summary.scalar("epsilon", epsilon, step=cur_frame)

            if cur_frame == 100 and profile:
                with summary_writer.as_default():
                    tf.summary.trace_export(
                        name="trace", step=cur_frame, profiler_outdir=log_dir
                    )
            if cur_frame % 100 == 0:
                end = time.time()
                print(end-start)
                start = time.time()

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
    print("| Env: {:>23s}   |".format(env_name))
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

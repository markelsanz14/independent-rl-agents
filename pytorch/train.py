import time
import argparse

import gym
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# from torch.utils.data import DataLoader

from envs import ATARI_ENVS
from atari_wrappers import make_atari, wrap_deepmind
from replay_buffers.uniform import UniformBuffer, DatasetBuffer

from networks.nature_cnn import NatureCNN
from networks.dueling_cnn import DuelingCNN

# from agents.ddpg import DDPG
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
    batch_size=32,
    initial_exploration=1.0,
    final_exploration=0.01,
    exploration_steps=int(2e6),
    learning_starts=50,  # int(1e4),
    train_freq=1,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if env_name in ATARI_ENVS:
        env = make_atari(env_name)
        env = wrap_deepmind(env, frame_stack=True, scale=False, clip_rewards=False)
    else:
        env = gym.make(env_name)

    if isinstance(env.action_space, gym.spaces.Discrete):
        num_state_feats = env.observation_space.shape
        num_actions = env.action_space.n

        replay_buffer = UniformBuffer(size=buffer_size, device=device)
        #dataset = DatasetBuffer(size=buffer_size, device=device)
        #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True)
        #iterator = iter(dataloader)
        # replay_buffer = DatasetBuffer(size=buffer_size, device=device)
        # buffer_loader =
        # DataLoader(replay_buffer, batch_size=batch_size, num_workers=0)
        if dueling:
            main_network = DuelingCNN(num_actions).to(device)
            target_network = DuelingCNN(num_actions).to(device)
        else:
            main_network = NatureCNN(num_actions).to(device)
            target_network = NatureCNN(num_actions).to(device)
        # main_network.apply(NatureCNN.init_weights)

        target_network.load_state_dict(main_network.state_dict())
        target_network.eval()

        agent = agent_class(
            env_name,
            num_actions=num_actions,
            main_nn=main_network,
            target_nn=target_network,
            device=device,
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
            device=device,
        )

    print_env_info(env, env_name, agent, main_network)

    # Creaete TensorBoard Metrics.
    log_dir = "logs/{}_{}_{}_ClipRew{}".format(
        agent_class.__name__, type(main_network).__name__, env_name, clip_rewards
    )
    writer = SummaryWriter(log_dir)
    # writer.add_graph(
    #    main_network, torch.randn(1, 4, 84, 84, dtype=torch.float32, device=device)
    # )

    imp = np.array([1.0 for _ in range(batch_size)])
    beta = 0.7

    epsilon = initial_exploration
    returns = []
    cur_frame, episode = 0, 0
    start = time.time()

    # Start learning!
    while cur_frame <= num_steps:
        state = env.reset()
        done, ep_rew, clip_ep_rew = False, 0, 0
        # Start an episode.
        while not done:
            state_np = np.expand_dims(state, axis=0).transpose(0, 3, 2, 1)
            state_in = torch.from_numpy(state_np).to(device, non_blocking=True)
            if normalize_obs:
                state_in = torch.div(state_in, 255.0)
            # Sample action from policy and take that action in the env.
            action = agent.take_exploration_action(state_in, env, epsilon)
            next_state, rew, done, info = env.step(action)
            if clip_rewards:
                reward = np.sign(rew)
            replay_buffer.add(state, action, reward, next_state, done)

            cur_frame += 1
            clip_ep_rew += reward
            ep_rew += rew
            state = next_state

            if cur_frame > learning_starts and cur_frame % train_freq == 0:
                # Perform training on data sampled from the replay buffer.
                if prioritized:
                    try: 
                        st, act, rew, next_st, d, imp, indx = iterator.next()#replay_buffer.sample(batch_size)
                    except StopIteration:
                        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True)
                        iterator = iter(dataloader)
                else:
                    # if cur_frame % 1000 == 0:
                    #    replay_buffer.shuffle_data()
                    # st, act, rew, next_st, d, idx = next(iter(buffer_loader))
                    try:
                        st, act, rew, next_st, d = replay_buffer.sample(batch_size)
                        #st, act, rew, next_st, d = iterator.next()#replay_buffer.sample(batch_size)
                        #st = st.to(device, non_blocking=True)
                        #act = act.to(device, non_blocking=True)
                        #rew = rew.to(device, non_blocking=True)
                        #next_st = next_st.to(device, non_blocking=True)
                        #d = d.to(device, non_blocking=True)
                    except StopIteration:
                        iterator = iter(dataloader)
                        continue
                    beta = 0.0
                if normalize_obs:
                    st = torch.div(st, 255.0).to(device)
                    next_st = torch.div(next_st, 255.0).to(device)
                loss_tuple = agent.train_step(st, act, rew, next_st, d, imp ** beta)
                if prioritized:
                    # Update priorities
                    # replay_buffer.update_priorities(indx, td_errors)
                    pass
            if cur_frame % 100 == 0:
                end = time.time()
                print(end-start)
                start = time.time()

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
                agent.target_nn.load_state_dict(agent.main_nn.state_dict())

            if cur_frame % save_ckpt_freq == 0 and cur_frame > learning_starts:
                agent.save_checkpoint()

            # Add TensorBoard Summaries.
            if cur_frame % 100000 == 0:
                writer.add_scalar("epsilon", epsilon, cur_frame)
        writer.add_scalar("return", ep_rew, episode)
        writer.add_scalar("clipped_return", clip_ep_rew, episode)

        episode += 1
        returns.append(ep_rew)

        if episode % 100 == 0:
            print_result(env_name, epsilon, episode, cur_frame, returns)
            returns = []
    agent.save_checkpoint()


def decay_epsilon(epsilon, step, initial_exp, final_exp, exp_steps):
    if step < exp_steps:
        epsilon -= (initial_exp - final_exp) / float(exp_steps)
    return epsilon


def print_result(env_name, epsilon, episode, step, returns):
    print("-------------------------------")
    print("| Env: {:>20s}   |".format(env_name[:-14]))
    print("| Exploration time %: {:>4d}%   |".format(int(epsilon * 100)))
    print("| Episode: {:>16d}   |".format(episode))
    print("| Steps: {:>18d}   |".format(step))
    print("| Last 100 return: {:>8.2f}   |".format(np.mean(returns)))
    print("-------------------------------")


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

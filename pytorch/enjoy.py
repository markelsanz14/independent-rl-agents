import os
import argparse

import numpy as np
import torch
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
    parser.add_argument("--clip_rewards", type=bool, default=False)
    parser.add_argument("--num_episodes", type=int, default=10)

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

    enjoy_env(
        env_name=args.env,
        agent_class=agent_class,
        num_episodes=args.num_episodes,
    )


def enjoy_env(
    env_name,
    agent_class,
    epsilon=0.01,
    prioritized=False,
    clip_rewards=False,
    num_episodes=20,
):
    """Runs an agent in a single environment to evaluate its performance.
    Args:
        env_name: str, name of a gym environment.
        agent_class: class object, one of the agents in the agent directory.
        prioritized: bool, whether to use prioritized experience replay.
        clip_rewards: bool, whether to clip the rewards to {-1, 0, 1} or not.
    """
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

    for episode in range(num_episodes):
        state = env.reset()
        done, ep_rew = False, 0
        while not done:
            env.render()
            state_in = torch.tensor(
                np.expand_dims(state, axis=0), dtype=torch.float32
            ).transpose(1, 3)
            # Sample action from policy and take that action in the env.
            action = agent.take_exploration_action(state_in, env, epsilon)
            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_rew += reward

        print("Epiosde {} return: {}".format(episode, ep_rew))


if __name__ == "__main__":
    main()

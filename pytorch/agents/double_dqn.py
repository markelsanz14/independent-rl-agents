"""Implements the Double DQN algorithm."""
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class DoubleDQN(object):
    """Implements the Double DQN algorithm and some helper methods."""

    def __init__(
        self,
        env_name,
        num_actions,
        main_nn,
        target_nn,
        lr=1e-5,
        discount=0.99,
        device="cpu",
    ):
        """Initializes the class.
        Args:
            env_name: str, id that identifies the environment in OpenAI gym.
            num_actions: int, number of discrete actions for the environment.
            main_nn: torch.nn.Module, a neural network from the ../networks/* directory.
            target_nn: torch.nn.Module, a neural network with the same architecture as main_nn.
            lr: float, a learning rate for the optimizer.
            discount: float, the discount factor for the Bellman equation.
            device: the result of running torch.device().
        """
        self.num_actions = num_actions
        self.discount = discount
        self.main_nn = main_nn
        self.target_nn = target_nn

        self.optimizer = optim.Adam(self.main_nn.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss.

        self.save_path = "./saved_models/{}-DoubleDQN-{}.pt".format(env_name, type(main_nn).__name__)
        if os.path.isfile(self.save_path):
            self.main_nn = torch.load(self.save_path)
            print("Loaded model from {}:".format(self.save_path))
        else:
            print("Initializing main neural network from scratch.")

    def save_checkpoint(self):
        """Saves the NN in the corresponding path."""
        torch.save(self.main_nn, self.save_path)
        print("Saved main_nn at {}".format(self.save_path))

    def take_exploration_action(self, state, env, epsilon=0.1):
        """Take random action with probability epsilon, else take best action.
        Args:
            state: tf.Tensor, state to be passed as input to the NN.
            env: gym.Env, gym environemnt to be used to take a random action
                sample.
            epsilon: float, value in range [0, 1] to define the probability of
                taking a random action in the epsilon-greedy policy.
        Returns:
            action: int, the action number that was selected.
        """
        result = np.random.uniform()
        if result < epsilon:
            return env.action_space.sample()
        else:
            q = self.main_nn(state).cpu().data.numpy()
            return np.argmax(q)  # Greedy action for state

    def train_step(self, states, actions, rewards, next_states, dones, importances):
        """Perform a training iteration on a batch of data.
        Args:
            states: Tensor, batch of states to be passed to the network.
            actions: Tensor, batch of actions (not one-hot encoded).
            rewards: Tensor, batch of rewards.
            next_states: Tensor, batch of next states to be passed to the network.
            dones: Tensor, batch of {0., 1.} floats that indicate whether the states are terminal.
        Returns:
            loss: float, the loss value of the current training step.
        """
        # Calculate targets.
        next_qs_argmax = self.main_nn(next_states).argmax(dim=-1, keepdim=True)
        masked_next_qs = self.target_nn(next_states).gather(1, next_qs_argmax).squeeze()
        target = rewards + (1.0 - dones) * self.discount * masked_next_qs
        masked_qs = self.main_nn(states).gather(1, actions.unsqueeze(dim=-1)).squeeze()
        loss = self.loss_fn(masked_qs, target.detach())
        # nn.utils.clip_grad_norm_(loss, max_norm=10)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return (loss,)

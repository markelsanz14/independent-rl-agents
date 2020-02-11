import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class DoubleDQN(object):
    """Implement the Double DQN algorithm and some helper methods."""

    def __init__(
        self,
        env_name,
        num_actions,
        main_nn,
        target_nn,
        replay_buffer,
        lr=1e-5,
        discount=0.99,
        batch_size=32,
        device="cpu",
    ):
        """Initializes the class."""
        self.num_actions = num_actions
        self.discount = discount
        self.main_nn = main_nn
        self.target_nn = target_nn
        self.buffer = replay_buffer

        self.optimizer = optim.Adam(self.main_nn.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss.

        self.save_path = "./saved_models/DQN-{}.pt".format(env_name)
        if os.path.isfile(self.save_path):
            self.main_nn = torch.load(self.save_path)
            print("Loaded model from {}:".format(self.save_path))
        else:
            print("Initializing main neural network from scratch.")

    def save_checkpoint(self, step):
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
            q = self.main_nn(state).data.numpy()
            return np.argmax(q)  # Greedy action for state

    def train_step(
        self, states, actions, rewards, next_states, dones, importances
    ):
        """Perform a training iteration on a batch of data sampled from the experience
        replay buffer.
        Returns:
            loss: float, the loss value of the current training step.
        """
        # Calculate targets.
        with torch.no_grad():
            next_qs_argmax = self.main_nn(next_states).argmax(dim=-1)
            next_action_mask = F.one_hot(next_qs_argmax, self.num_actions)
            next_qs_target = self.target_nn(next_states)
            masked_next_qs = (next_action_mask * next_qs_target).sum(dim=-1)
            target = rewards + (1.0 - dones) * self.discount * masked_next_qs
        qs = self.main_nn(states)
        action_masks = F.one_hot(actions, self.num_actions)
        masked_qs = (action_masks * qs).sum(dim=-1)
        td_errors = (target - masked_qs).abs()
        loss = self.loss_fn(masked_qs, target)  # sample_weight=importances
        nn.utils.clip_grad_norm_(loss, max_norm=10)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return (loss,), td_errors

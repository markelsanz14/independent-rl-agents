"""Implements the DQN algorithm, with the Neural Network and Replay Buffer."""
import os
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from envs import ATARI_ENVS  # Remove this line if you only need this file.


class ReplayBuffer(object):
    """Experience replay buffer that samples uniformly."""

    def __init__(self, size):
        """Initializes the buffer."""
        self.buffer = deque(maxlen=size)

    def add_to_buffer(self, state, action, reward, next_state, done):
        """Adds data to experience replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        """Samples num_sample elements from the buffer."""
        batch = random.sample(self.buffer, num_samples)
        states, actions, rewards, next_states, dones = list(zip(*batch))
        states = torch.tensor(states, dtype=torch.float32).transpose(1, 3)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32).transpose(
            1, 3
        )
        dones = torch.tensor(dones, dtype=torch.float32)
        return states, actions, rewards, next_states, dones


class PrioritizedReplayBuffer(object):
    """Experience replay buffer that samples proportionately to the TD errors
    for each sample. See the paper for details.
    This version uses a list and not a sum tree, so they performance of this
    version is not optimal for big replay buffers.
    """

    def __init__(self, size: int, alpha: float = 0.6):
        """Initializes the buffer.
        Args:
            size: int, the max size of the replay buffer.
            alpha: float, the strength of the prioritization
                (0.0 - no prioritization, 1.0 - full prioritization).
        """
        self.buffer = deque(maxlen=size)
        self.priorities = deque(maxlen=size)
        self._alpha = alpha

    def __len__(self):
        return len(self.buffer)

    def add_to_buffer(self, state, action, reward, next_state, done):
        """Adds data to experience replay buffer with priority."""
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max(self.priorities, default=1.0))

    def calculate_probabilities(self):
        """Calculates probability of being sampled for each element in the buffer.
        Returns:
            probabilities: np.array, a list of values summing up to 1.0, with
                the probability of each element in the buffer being sampled
                according to its priority.
        """
        priorities = np.array(self.priorities) ** self._alpha
        return priorities / sum(priorities)

    def calculate_importance(self, probs):
        """Calculates the importance sampling bias correction."""
        importance = 1.0 / len(self.buffer) * 1.0 / probs
        return importance / max(importance)

    def sample(self, num_samples):
        """Samples num_sample elements from the buffer."""
        probs = self.calculate_probabilities()
        batch_indices = random.choices(
            range(len(self.buffer)), k=num_samples, weights=probs
        )
        batch = np.array(self.buffer)[batch_indices]
        importance = self.calculate_importance(probs[batch_indices])
        states, actions, rewards, next_states, dones = list(zip(*batch))
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            importance,
            batch_indices,
        )

    def update_priorities(self, indices, td_errors, offset=0.01):
        """Updates the priorities for a batch given the TD errors.
        Args:
            indices: np.array, the list of indices of the priority list to be
                updated.
            td_errors: np.array, the list of TD errors for those indices. The
                priorities will be updated to the TD errors plus the offset.
            offset: float, small positive value to ensure that all trajectories
                have some probability of being selected.
        """
        for index, error in zip(indices, td_errors):
            self.priorities[index] = error + offset


class QNetworkConv(nn.Module):
    """Convolutional neural network for the Atari games."""

    def __init__(self, num_actions):
        """Initializes the neural network."""
        super(QNetworkConv, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv1size = conv2d_size_out(84, kernel_size=8, stride=4)
        conv2size = conv2d_size_out(conv1size, kernel_size=4, stride=2)
        conv3size = conv2d_size_out(conv2size, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * conv3size * conv3size, 512)
        self.out = nn.Linear(512, num_actions)

    def forward(self, states):
        """Forward pass of the neural network with some inputs.
        Args:
            states: Tensor, batch of states.
        Returns:
            qs: Tensor, the q-values of the given state for all possible
                actions.
        """
        x = F.relu(self.conv1(states))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))  # Flatten input.
        return self.out(x)


class QNetwork(nn.Module):
    """Dense neural network for simple games."""

    def __init__(self, num_inputs, num_actions):
        """Initializes the neural network."""
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, num_actions)

    def forward(self, states):
        """Forward pass of the neural network with some inputs.
        Args:
            states: Tensor, batch of states.
        Returns:
            qs: Tensor, the q-values of the given state for all possible
                actions.
        """
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        return self.out(x)


class DQN(object):
    """Implement the DQN algorithm and some helper methods."""

    def __init__(
        self,
        env_name,
        num_state_feats,
        num_actions,
        prioritized=False,
        prioritization_alpha=0.6,
        lr=1e-4,
        buffer_size=100000,
        discount=0.99,
        device="cpu",
    ):
        """Initializes the class."""
        self.num_state_feats = num_state_feats
        self.num_actions = num_actions
        self.discount = discount
        if prioritized:
            self.buffer = PrioritizedReplayBuffer(
                size=buffer_size, alpha=prioritization_alpha
            )
        else:
            self.buffer = ReplayBuffer(buffer_size)

        if env_name in ["CarRacing-v0"] + ATARI_ENVS:
            self.main_nn = QNetworkConv(num_actions).to(device)
            self.target_nn = QNetworkConv(num_actions).to(device)
        else:
            self.main_nn = QNetwork(num_actions).to(device)
            self.target_nn = QNetwork(num_actions).to(device)

        self.optimizer = optim.Adam(self.main_nn.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss.

        self.save_path = "./saved_models/DQN-{}.pt".format(env_name)
        if os.path.isfile(self.save_path):
            self.main_nn = torch.load(self.save_path)
            print("Loaded model from {}:".format(self.save_path))

    def save_checkpoint(self, step):
        torch.save(self.main_nn, self.save_path)
        print(
            "Saved main_nn checkpoint for step {}: {}".format(
                step, self.save_path
            )
        )

    def take_exploration_action(self, state, env, epsilon=0.1):
        """Take random action with probability epsilon, else take best action.
        Args:
            state: Tensor, state to be passed as input to the NN.
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
            max_next_qs = self.target_nn(next_states).max(-1).values
            target = rewards + (1.0 - dones) * self.discount * max_next_qs
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

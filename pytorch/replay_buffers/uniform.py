import random
from collections import deque
import numpy as np
import torch

    
class UniformBuffer(object):
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
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)
        for i in idx:
            elem = self.buffer[i]
            state, action, reward, next_state, done = elem
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        states = torch.from_numpy(np.array(states)).transpose(1, 3)
        actions = torch.from_numpy(np.array(actions))
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32))
        next_states = torch.from_numpy(np.array(next_states)).transpose(1, 3)
        dones = torch.from_numpy(np.array(dones, dtype=np.float32))
        return states, actions, rewards, next_states, dones


class DatasetBuffer(torch.utils.data.IterableDataset):
    def __init__(self, size):
        super(DatasetBuffer).__init__()
        self.buffer = deque(maxlen=size)

    def add_to_buffer(self, state, action, reward, next_state, done):
        """Adds data to experience replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        """Samples num_sample elements from the buffer."""
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)
        for i in idx:
            elem = self.buffer[i]
            state, action, reward, next_state, done = elem
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        states = torch.from_numpy(np.array(states)).transpose(1, 3)
        actions = torch.from_numpy(np.array(actions))
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32))
        next_states = torch.from_numpy(np.array(next_states)).transpose(1, 3)
        dones = torch.from_numpy(np.array(dones, dtype=np.float32))
        return states, actions, rewards, next_states, dones

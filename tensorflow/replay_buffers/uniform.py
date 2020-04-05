from collections import deque

import tensorflow as tf
import numpy as np


class UniformBuffer(object):
    """Experience replay buffer that samples uniformly."""
    __name__ = "UniformBuffer"

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
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)
        return states, actions, rewards, next_states, dones


class DatasetUniformBuffer(object):
    """Experience replay buffer that samples uniformly."""

    def __init__(self, size):
        """Initializes the buffer."""
        self.buffer = deque(maxlen=size)

    def add_to_buffer(self, state, action, reward, next_state, done):
        """Adds data to experience replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample_gen(self):
        """Samples num_sample elements from the buffer."""
        while True:
            states, actions, rewards, next_states, dones = [], [], [], [], []
            idx = np.random.choice(len(self.buffer), 1)
            elem = self.buffer[idx[0]]
            state, action, reward, next_state, done = elem
            states = np.array(state, copy=False)
            actions = np.array(action, copy=False)
            rewards = np.array(reward, dtype=np.float32)
            next_states = np.array(next_state, copy=False)
            dones = np.array(done, dtype=np.float32)
            yield states, actions, rewards, next_states, dones

    def build_iterator(self, batch_size):
        dataset = tf.data.Dataset.from_generator(
            self.sample_gen, (tf.uint8, tf.int32, tf.float32, tf.uint8, tf.float32)
        )
        dataset = dataset.batch(batch_size).prefetch(3)
        return iter(dataset)

"""Class that implements a uniform experience replay buffer."""
import tensorflow as tf
import numpy as np


class UniformBuffer(object):
    """Experience replay buffer that samples uniformly."""
    __name__ = "UniformBuffer"

    def __init__(self, size):
        """Initializes the buffer."""
        self._size = size
        self.buffer = []
        self._next_idx = 0

    def add(self, state, action, reward, next_state, done):
        """Adds data to experience replay buffer."""
        if self._next_idx >= len(self.buffer):
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self._next_idx] = (state, action, reward, next_state, done)
        self._next_idx = (self._next_idx + 1) % self._size

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
    __name__ = "DatasetUniformBuffer"

    def __init__(self, size, normalization_val=1.):
        """Initializes the buffer."""
        self.buffer = []
        self._size = size
        self.normalization_val = normalization_val
        self._next_idx = 0

    def add(self, state, action, reward, next_state, done):
        """Adds data to experience replay buffer."""
        if self._next_idx >= len(self.buffer):
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self._next_idx] = (state, action, reward, next_state, done)
        self._next_idx = (self._next_idx + 1) % self._size

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
    
    #def normalize_obs(self, state, action, reward, next_state, done):
    #    state_ = tf.cast(state, tf.float32) / self.normalization_val
    #    next_state_ = tf.cast(next_state, tf.float32) / self.normalization_val
    #    return state_, action, reward, next_state_, done

    def build_iterator(self, batch_size):
        dataset = tf.data.Dataset.from_generator(
                generator=self.sample_gen,
                output_types=(tf.uint8, tf.int32, tf.float32, tf.uint8, tf.float32),
                output_shapes=((84, 84, 4), (), (), (84, 84, 4), ()),
        )
        #dataset = dataset.map(self.normalize_obs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        #dataset = dataset.shuffle(buffer_size=100, reshuffle_each_iteration=True)
        #dataset = dataset.repeat(-1)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))
        return iter(dataset)

import random
from itertools import cycle, islice
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

    @profile
    def add_to_buffer(self, state, action, reward, next_state, done):
        """Adds data to experience replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def __get_item__(self, index):
        return self.buffer[index]

    @profile
    def sample_gen(self, start, end):
        while True:
            states, actions, rewards, next_states, dones = [], [], [], [], []
            #idx = np.random.choice(end-start, self.batch_size // num_workers) + start
            i = np.random.randint(end-start) + start
            elem = self.buffer[i]
            state, action, reward, next_state, done = elem
            states = torch.from_numpy(np.array(state, copy=False)).transpose(0, 2)
            actions = torch.from_numpy(np.array(action, copy=False))
            rewards = torch.from_numpy(np.array(reward, dtype=np.float32))
            next_states = torch.from_numpy(np.array(next_state, copy=False)).transpose(0, 2)
            dones = torch.from_numpy(np.array(done, dtype=np.float32))
            yield states, actions, rewards, next_states, dones, i

    def shuffle_data(self):
        self.buffer = random.sample(self.buffer, k=len(self.buffer))

    @profile
    def sample_gen2(self, start, end, worker_id, num_workers):
        #self.idx = np.random.randint(end-start, size=end-start) + start
        for index, elem in enumerate(islice(self.buffer, worker_id, len(self.buffer), num_workers)):
            #elem = self.buffer[i]
            state, action, reward, next_state, done = elem
            states = torch.from_numpy(np.array(state, copy=False)).transpose(0, 2)
            actions = torch.from_numpy(np.array(action, copy=False))
            rewards = torch.from_numpy(np.array(reward, dtype=np.float32))
            next_states = torch.from_numpy(np.array(next_state, copy=False)).transpose(0, 2)
            dones = torch.from_numpy(np.array(done, dtype=np.float32))
            yield states, actions, rewards, next_states, dones, index

    @profile
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single worker.
            start = 0
            end = len(self.buffer)
        else:  # Inside a worker process.
            per_worker = len(self.buffer) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker
        return cycle(self.sample_gen2(start, end, worker_info.id, worker_info.num_workers))

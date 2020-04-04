import random
from itertools import cycle, islice
from collections import deque
import numpy as np
import torch


class UniformBuffer(object):
    """Experience replay buffer that samples uniformly."""

    def __init__(self, size, device="cpu"):
        """Initializes the buffer."""
        #self.buffer = deque(maxlen=size)
        self._size = size
        self.buffer = []
        self.device = device
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
        states = torch.as_tensor(
            np.array(states).transpose(0, 3, 2, 1), device=self.device
        )
        actions = torch.as_tensor(np.array(actions), device=self.device)
        rewards = torch.as_tensor(
            np.array(rewards, dtype=np.float32), device=self.device
        )
        next_states = torch.as_tensor(
            np.array(next_states).transpose(0, 3, 2, 1), device=self.device
        )
        dones = torch.as_tensor(np.array(dones, dtype=np.float32), device=self.device)
        return states, actions, rewards, next_states, dones


class DatasetBuffer(torch.utils.data.Dataset):
    def __init__(self, size, device):
        super(DatasetBuffer).__init__()
        self.buffer = [(np.random.randint(255, size=(84, 84, 4), dtype=np.uint8), 0, 0, np.random.randint(255, size=(84, 84, 4), dtype=np.uint8), 0)]
        print("Buffer initialized.")
        self._size = size
        self.device = device
        self._next_idx = 0
        self._len = 1

    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        st, act, rew, n_st, d =  self.buffer[idx]
        state = torch.from_numpy(np.array(st).transpose(2, 1, 0))
        action = torch.from_numpy(np.array(act))
        reward = torch.from_numpy(np.array(rew, dtype=np.float32))
        next_state = torch.from_numpy(np.array(n_st).transpose(2, 1, 0))
        done = torch.from_numpy(np.array(d, dtype=np.float32))
        return state, action, reward, next_state, done

    def add(self, state, action, reward, next_state, done):
        if self._next_idx >= len(self.buffer):
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self._next_idx] = (state, action, reward, next_state, done)
        self._next_idx = (self._next_idx + 1) % self._size

        if self._len < len(self.buffer):#self._size:
            self._len += 1


class IterDatasetBuffer(torch.utils.data.IterableDataset):
    def __init__(self, size, device="cpu"):
        super(IterDatasetBuffer).__init__()
        self.buffer = deque(maxlen=size)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Adds data to experience replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def __get_item__(self, index):
        return self.buffer[index]

    def sample_gen(self, start, end):
        while True:
            states, actions, rewards, next_states, dones = [], [], [], [], []
            # idx = np.random.choice(end-start, self.batch_size // num_workers) + start
            i = np.random.randint(end - start) + start
            elem = self.buffer[i]
            state, action, reward, next_state, done = elem
            states = torch.from_numpy(np.array(state, copy=False)).transpose(0, 2)
            actions = torch.from_numpy(np.array(action, copy=False))
            rewards = torch.from_numpy(np.array(reward, dtype=np.float32))
            next_states = torch.from_numpy(np.array(next_state, copy=False)).transpose(
                0, 2
            )
            dones = torch.from_numpy(np.array(done, dtype=np.float32))
            yield states, actions, rewards, next_states, dones, i

    def shuffle_data(self):
        self.buffer = random.sample(self.buffer, k=len(self.buffer))

    def sample_gen2(self, start, end, worker_id, num_workers):
        # self.idx = np.random.randint(end-start, size=end-start) + start
        for index, elem in enumerate(
            islice(self.buffer, worker_id, len(self.buffer), num_workers)
        ):
            # elem = self.buffer[i]
            state, action, reward, next_state, done = elem
            states = torch.as_tensor(
                np.array(state, copy=False), device=self.device
            ).transpose(0, 2)
            actions = torch.as_tensor(np.array(action, copy=False), device=self.device)
            rewards = torch.as_tensor(
                np.array(reward, dtype=np.float32), device=self.device
            )
            next_states = torch.as_tensor(
                np.array(next_state, copy=False), device=self.device
            ).transpose(0, 2)
            dones = torch.as_tensor(
                np.array(done, dtype=np.float32), device=self.device
            )
            yield states, actions, rewards, next_states, dones, index

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single worker.
            worker_id = 0
            start = 0
            end = len(self.buffer)
            num_workers = 1
        else:  # Inside a worker process.
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            per_worker = len(self.buffer) // num_workers
            start = worker_id * per_worker
            end = start + per_worker
        return cycle(self.sample_gen2(start, end, worker_id, num_workers))

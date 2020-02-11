import random

from collections import deque
import numpy as np


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

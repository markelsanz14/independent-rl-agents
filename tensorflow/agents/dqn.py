"""Implements the DQN algorithm, with the Neural Network and Replay Buffer."""
import random
from collections import deque

import tensorflow as tf
import numpy as np

from envs import ATARI_ENVS  # Remove this line if you only need this file.


layers = tf.keras.layers


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
        states = np.array(states, copy=False, dtype=np.float32)
        actions = np.array(actions, copy=False)
        rewards = np.array(rewards, copy=False, dtype=np.float32)
        next_states = np.array(next_states, copy=False, dtype=np.float32)
        dones = np.array(dones, copy=False, dtype=np.float32)
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
            self.priorities[index] = abs(error) + offset


class QNetworkConv(tf.keras.Model):
    """Convolutional neural network for the Atari games."""

    def __init__(self, num_actions):
        """Initializes the neural network."""
        super(QNetworkConv, self).__init__()
        self.conv1 = layers.Conv2D(
            filters=32,
            kernel_size=8,
            strides=4,
            activation="relu",
            kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
            bias_initializer=tf.keras.initializers.Zeros(),
        )
        self.conv2 = layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            activation="relu",
            kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
            bias_initializer=tf.keras.initializers.Zeros(),
        )
        self.conv3 = layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            activation="relu",
            kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
            bias_initializer=tf.keras.initializers.Zeros(),
        )
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(
            units=512,
            activation="relu",
            kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
            bias_initializer=tf.keras.initializers.Zeros(),
        )
        self.out = layers.Dense(units=num_actions)

    @tf.function
    def call(self, states):
        """Forward pass of the neural network with some inputs.
        Args:
            states: tf.Tensor, batch of states.
        Returns:
            qs: tf.Tensor, the q-values of the given state for all possible
                actions.
        """
        x = self.conv1(states)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.out(x)


class QNetwork(tf.keras.Model):
    """Dense neural network for simple games."""

    def __init__(self, num_actions):
        """Initializes the neural network."""
        super(QNetwork, self).__init__()
        self.dense1 = layers.Dense(32, activation="relu")
        self.dense2 = layers.Dense(32, activation="relu")
        self.out = layers.Dense(num_actions)

    def call(self, states):
        """Forward pass of the neural network with some inputs.
        Args:
            states: tf.Tensor, batch of states.
        Returns:
            qs: tf.Tensor, the q-values of the given state for all possible
                actions.
        """
        x = self.dense1(states)
        x = self.dense2(x)
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
        lr=1e-5,
        buffer_size=100000,
        discount=0.99,
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
            self.main_nn = QNetworkConv(num_actions)
            self.target_nn = QNetworkConv(num_actions)
        else:
            self.main_nn = QNetwork(num_actions)
            self.target_nn = QNetwork(num_actions)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, clipnorm=10
        )
        self.loss = tf.keras.losses.Huber()

        # Checkpoints.
        self.ckpt_main = tf.train.Checkpoint(
            step=tf.Variable(1), optimizer=self.optimizer, net=self.main_nn
        )
        self.manager_main = tf.train.CheckpointManager(
            self.ckpt_main,
            "./saved_models/DQN-{}".format(env_name),
            max_to_keep=3,
        )
        self.ckpt_main.restore(
            self.manager_main.latest_checkpoint
        ).expect_partial()
        if self.manager_main.latest_checkpoint:
            print(
                "Restored from {}".format(self.manager_main.latest_checkpoint)
            )
        else:
            print("Initializing main neural network from scratch.")

    def save_checkpoint(self):
        save_path_main = self.manager_main.save()
        print(
            "Saved main_nn checkpoint for step {}: {}".format(
                int(self.ckpt_main.step), save_path_main
            )
        )

    @tf.function
    def run_main_nn(self, state):
        """Compiles the call to the main_nn into a tf graph."""
        return self.main_nn(state)

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
            q = self.run_main_nn(state).numpy()
            return np.argmax(q)  # Greedy action for state

    @tf.function
    def train_step(
        self, states, actions, rewards, next_states, dones, importances
    ):
        """Perform a training iteration on a batch of data sampled from the experience
        replay buffer.
        Returns:
            loss: float, the loss value of the current training step.
        """
        # Calculate targets.
        next_qs = self.target_nn(next_states)
        max_next_qs = tf.reduce_max(next_qs, axis=-1)
        target = rewards + (1.0 - dones) * self.discount * max_next_qs
        with tf.GradientTape() as tape:
            qs = self.main_nn(states)
            action_masks = tf.one_hot(actions, self.num_actions)
            masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
            td_errors = target - masked_qs
            loss = self.loss(
                tf.stop_gradient(target), masked_qs
            )  # , sample_weight=importances)
        grads = tape.gradient(loss, self.main_nn.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.main_nn.trainable_variables)
        )

        return (loss,), td_errors

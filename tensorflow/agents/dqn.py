"""Implements the DQN algorithm."""
import random

import tensorflow as tf
import numpy as np

# Modify these lines if you only need this file:
from envs import ATARI_ENVS
from replay_buffers.uniform import UniformBuffer
from networks.nature_cnn import NatureCNN


class DQN(object):
    """Implement the DQN algorithm and some helper methods."""

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
    ):
        """Initializes the class."""
        self.num_actions = num_actions
        self.discount = discount
        self.main_nn = main_nn
        self.target_nn = target_nn
        self.buffer = replay_buffer

        self.optimizer = tf.keras.optimizers.Adam(lr, clipnorm=10)
        self.loss = tf.keras.losses.Huber()

        # Checkpoints.
        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(1), optimizer=self.optimizer, net=self.main_nn
        )
        self.manager = tf.train.CheckpointManager(
            self.ckpt,
            "./saved_models/{}-DQN-{}".format(
                env_name, type(main_nn).__name__
            ),
            max_to_keep=3,
        )
        self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing main neural network from scratch.")

    def save_checkpoint(self):
        path = self.manager.save()
        print("Saved main_nn at {}".format(path))

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
            return env.action_space.sample()  # Random action.
        else:
            state = np.expand_dims(state, axis=0)
            q = self.main_nn(state).numpy()
            return np.argmax(q)  # Greedy action.

    @tf.function
    def train_step(
        self, states, actions, rewards, next_states, dones, importances=None
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

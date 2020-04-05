"""Implements the Soft Actor-Critic algorithm."""
import tensorflow as tf
import numpy as np


class SAC(object):
    """Implement the Soft Actor-Critic algorithm and some helper methods."""
    __name__ = "SAC"

    def __init__(
        self,
        env_name,
        main_v_nn,
        target_v_nn,
        q1_nn,
        q2_nn,
        policy_nn,
        replay_buffer=None,
        lr=1e-5,
        discount=0.99,
        batch_size=32,
    ):
        """Initializes the class."""
        self.main_v_nn = main_v_nn
        self.target_v_nn = target_v_nn
        self.q1_nn = q1_nn
        self.q2_nn = q2_nn
        self.policy_nn = policy_nn

        self.optimizer_v = tf.keras.optimizers.Adam(lr=lr, clipnorm=10)
        self.optimizer_q1 = tf.keras.optimizers.Adam(lr=lr, clipnorm=10)
        self.optimizer_q2 = tf.keras.optimizers.Adam(lr=lr, clipnorm=10)
        self.optimizer_policy = tf.keras.optimizers.Adam(lr=lr, clipnorm=10)
        self.loss_v_fn = tf.keras.losses.MeanSquaredError()
        self.loss_q1_fn = tf.keras.losses.MeanSquaredError()
        self.loss_q2_fn = tf.keras.losses.MeanSquaredError()
        self.discount = discount

        # Checkpoints.
        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(1), optimizer=self.optimizer_policy, net=policy_nn
        )
        self.manager = tf.train.CheckpointManager(
            self.ckpt,
            "./saved_models/{}-SAC-policy-{}".format(env_name, type(policy_nn).__name__),
            max_to_keep=1,
        )
        self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing policy network from scratch.")

    def save_checkpoint(self):
        path = self.manager.save()
        print("Saved policy_nn at {}".format(path))

    def take_exploration_action(self, state, env, noise=0.1):
        state_in = np.expand_dims(state, axis=0)
        pi, action, log_prob, mu, log_std = self.policy_nn(state_in)
        return action[0]

    def update_target_network(self, source_weights, target_weights, tau=0.005):
        """Updates target network using Polyak averaging.
        Args:
          source_weights: list, a list of the NN weights resulting of running
              nn.weights on keras. Weights to be copied from.
          target_weights: list, a list of the NN weights resulting of running
              nn.weights on keras. Weights to be copied to.
          tau: float, value in range [0, 1] for the polyak averaging value.
        """
        for source_weight, target_weight in zip(source_weights, target_weights):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)

    @tf.function
    def train_step(self, state, action, reward, next_state, done, importances=None):
        """Trains all networks on a batch of data."""
        next_state_v = self.target_v_nn(next_state)
        target = reward + (1. - done) * self.discount * next_state_v

        # Train First Q-Network.
        with tf.GradientTape() as tape_q1:
            q1 = self.q1_nn(state, action)
            loss_q1 = self.loss_q1_fn(target, q1)
        grads_q1 = tape_q1.gradient(loss_q1, self.q1_nn.trainable_variables)
        self.optimizer_q1.apply_gradients(zip(grads_q1, self.q1_nn.trainable_variables))

        # Train Second Q-Network.
        with tf.GradientTape() as tape_q2:
            q2 = self.q2_nn(state, action)
            loss_q2 = self.loss_q2_fn(target, q2)
        grads_q2 = tape_q2.gradient(loss_q2, self.q2_nn.trainable_variables)
        self.optimizer_q2.apply_gradients(zip(grads_q2, self.q2_nn.trainable_variables))

        # Train Policy Network.
        with tf.GradientTape() as tape_pi:
            pi, policy_action, log_prob, mean, log_std = self.policy_nn(state)
            min_q = tf.stop_gradient(tf.math.minimum(self.q1_nn(state, policy_action),
                                                     self.q2_nn(state, policy_action)))
            loss_pi = tf.reduce_mean(log_prob - tf.stop_gradient(min_q))
        grads_pi = tape_pi.gradient(loss_pi, self.policy_nn.trainable_variables)
        self.optimizer_policy.apply_gradients(zip(grads_pi, self.policy_nn.trainable_variables))

        # Train Value Network.
        with tf.GradientTape() as tape_v:
            v_value = self.main_v_nn(state)
            target_v = tf.stop_gradient(min_q - log_prob)
            loss_v = self.loss_v_fn(target_v, v_value)
        grads_v = tape_v.gradient(loss_v, self.main_v_nn.trainable_variables)
        self.optimizer_v.apply_gradients(zip(grads_v, self.main_v_nn.trainable_variables))

        # Update target network.
        self.update_target_network(self.main_v_nn.weights, self.target_v_nn.weights)
        return loss_q1, loss_q2, loss_pi, loss_v

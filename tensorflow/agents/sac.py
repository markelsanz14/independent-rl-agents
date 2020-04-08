"""Implements the Soft Actor-Critic algorithm."""
import tensorflow as tf
import numpy as np


class SAC(object):
    """Implement the Soft Actor-Critic algorithm and some helper methods."""

    __name__ = "SAC"

    def __init__(
        self,
        env_name,
        main_q1_nn,
        main_q2_nn,
        target_q1_nn,
        target_q2_nn,
        policy_nn,
        target_entropy=None,
        num_action_feats=None,
        lr_q1=3e-4,
        lr_q2=3e-4,
        lr_policy=3e-4,
        lr_alpha=3e-4,
        discount=0.99,
        batch_size=32,
    ):
        """Initializes the class."""
        self.main_q1_nn = main_q1_nn
        self.main_q2_nn = main_q2_nn
        self.target_q1_nn = target_q1_nn
        self.target_q2_nn = target_q2_nn
        self.policy_nn = policy_nn
        self.alpha = tf.Variable(1.0, trainable=True)
        self.target_entropy = self.default_target_entropy(
            target_entropy, num_action_feats
        )

        self.optimizer_q1 = tf.keras.optimizers.Adam(lr=lr_q1, clipnorm=10)
        self.optimizer_q2 = tf.keras.optimizers.Adam(lr=lr_q2, clipnorm=10)
        self.optimizer_policy = tf.keras.optimizers.Adam(lr=lr_policy, clipnorm=10)
        self.optimizer_alpha = tf.keras.optimizers.Adam(lr=lr_alpha, clipnorm=10)
        self.loss_q1_fn = tf.keras.losses.MeanSquaredError()
        self.loss_q2_fn = tf.keras.losses.MeanSquaredError()
        self.discount = discount

        # Checkpoints.
        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(1), optimizer=self.optimizer_policy, net=policy_nn
        )
        self.manager = tf.train.CheckpointManager(
            self.ckpt,
            "./saved_models/{}-SAC-policy-{}".format(
                env_name, type(policy_nn).__name__
            ),
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

    def default_target_entropy(self, target_entropy, num_action_feats):
        """Heuristic to calculate a good target entropy."""
        if target_entropy is not None:
            return target_entropy
        return -np.prod(num_action_feats) / 2.0

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
        _, next_action, next_log_prob, _, _ = self.policy_nn(next_state)
        next_min_q = tf.squeeze(
            tf.math.minimum(
                self.target_q1_nn(next_state, next_action),
                self.target_q2_nn(next_state, next_action),
            )
        )
        target = tf.stop_gradient(
            reward
            + (1.0 - done) * self.discount * (next_min_q - self.alpha * next_log_prob)
        )

        # Train First Q-Network.
        with tf.GradientTape() as tape_q1:
            q1 = tf.squeeze(self.main_q1_nn(state, action))
            loss_q1 = self.loss_q1_fn(target, q1)
        grads_q1 = tape_q1.gradient(loss_q1, self.main_q1_nn.trainable_variables)
        self.optimizer_q1.apply_gradients(
            zip(grads_q1, self.main_q1_nn.trainable_variables)
        )
        # Train Second Q-Network.
        with tf.GradientTape() as tape_q2:
            q2 = tf.squeeze(self.main_q2_nn(state, action))
            loss_q2 = self.loss_q2_fn(target, q2)
        grads_q2 = tape_q2.gradient(loss_q2, self.main_q2_nn.trainable_variables)
        self.optimizer_q2.apply_gradients(
            zip(grads_q2, self.main_q2_nn.trainable_variables)
        )

        # Train Policy Network.
        with tf.GradientTape() as tape_pi:
            _, policy_action, log_prob, _, _ = self.policy_nn(state)
            min_q = tf.math.minimum(
                self.main_q1_nn(state, policy_action),
                self.main_q2_nn(state, policy_action),
            )
            loss_pi = tf.reduce_sum(
                self.alpha * log_prob - tf.squeeze(tf.stop_gradient(min_q))
            )
        grads_pi = tape_pi.gradient(loss_pi, self.policy_nn.trainable_variables)
        self.optimizer_policy.apply_gradients(
            zip(grads_pi, self.policy_nn.trainable_variables)
        )

        # Train alpha.
        with tf.GradientTape() as tape_alpha:
            losses_alpha = -self.alpha * tf.stop_gradient(
                log_prob + self.target_entropy
            )
            loss_alpha = tf.reduce_mean(losses_alpha)
        grads_alpha = tape_alpha.gradient(loss_alpha, [self.alpha])
        self.optimizer_alpha.apply_gradients(zip(grads_alpha, [self.alpha]))

        # Update target network.
        self.update_target_network(self.main_q1_nn.weights, self.target_q1_nn.weights)
        self.update_target_network(self.main_q2_nn.weights, self.target_q2_nn.weights)
        # print({'loss_q1': loss_q1, 'loss_q2': loss_q2, 'loss_policy': loss_pi, 'loss_alpha': loss_alpha})
        # print(self.alpha)
        return {
            "loss_q1": loss_q1,
            "loss_q2": loss_q2,
            "loss_policy": loss_pi,
            "loss_alpha": loss_alpha,
            "alpha": self.alpha,
        }

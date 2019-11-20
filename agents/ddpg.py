import tensorflow as tf
import random

layers = tf.keras.layers


class ReplayBuffer(object):
    """Experience replay buffer that samples uniformly."""

    def __init__(self, size):
        """Initializes the buffer."""
        self.buffer = []
        self._size = size
        self._next_pos = 0

    def __len__(self):
        return len(self.buffer)

    def add_to_buffer(self, state, action, reward, next_state, done):
        """Adds data to experience replay buffer."""
        if len(self.buffer) < self._size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self._next_pos] = (
                state,
                action,
                reward,
                next_state,
                done,
            )
        self._next_pos = (self._next_pos + 1) % self._size

    def sample(self, num_samples):
        """Samples num_sample elements from the buffer."""
        batch = random.sample(self.buffer, num_samples)
        states, actions, rewards, next_states, dones = list(zip(*batch))
        return states, actions, rewards, next_states, dones


class CriticNetworkConv(tf.keras.Model):
    def __init__(self):
        super(CriticNetworkConv, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            32, 3, strides=(2, 2), activation="relu"
        )  # out 48,48,32
        self.conv2 = tf.keras.layers.Conv2D(
            32, 4, strides=(2, 2), activation="relu"
        )  # out 24,24,32
        self.conv3 = tf.keras.layers.Conv2D(
            64, 4, strides=(2, 2), activation="relu"
        )  # out 12,12,64
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs):
        """Calls the neural network with some inputs.
        Args:
            inputs: tuple of (states, actions).
        Returns:
            q: the q-value of Q(states, actions).
        """
        states, actions = inputs
        x = self.conv1(states)
        x = self.conv2(x)
        x = self.conv3(x)
        x = tf.concat([self.flatten(x), actions], axis=-1)
        x = self.dense1(x)
        q = self.out(x)
        return q


class ActorNetworkConv(tf.keras.Model):
    def __init__(self, num_actions, min_action_values, max_action_values):
        super(ActorNetworkConv, self).__init__()
        self.min_action_values = min_action_values
        self.max_action_values = max_action_values

        self.conv1 = tf.keras.layers.Conv2D(
            32, 3, strides=(2, 2), activation="relu"
        )  # out 48,48,32
        self.conv2 = tf.keras.layers.Conv2D(
            32, 4, strides=(2, 2), activation="relu"
        )  # out 24,24,32
        self.conv3 = tf.keras.layers.Conv2D(
            64, 4, strides=(2, 2), activation="relu"
        )  # out 12,12,64
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.out = tf.keras.layers.Dense(num_actions, activation="tanh")

    def call(self, states):
        x = self.conv1(states)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.out(x)
        action = x * self.max_action_values
        action_clipped = tf.clip_by_value(
            action, self.min_action_values, self.max_action_values
        )
        return action_clipped


class CriticNetwork(tf.keras.Model):
    def __init__(self, num_state_feats, num_action_feats):
        super(CriticNetwork, self).__init__()
        self.dense1 = layers.Dense(400, activation="relu")
        self.dense2 = layers.Dense(300, activation="relu")
        self.out = layers.Dense(1)

    def call(self, inputs):
        states, actions = inputs
        x = tf.concat([states, actions], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        q = self.out(x)
        return q


class ActorNetwork(tf.keras.Model):
    def __init__(self, num_state_feats, num_action_feats, max_action_values):
        super(ActorNetwork, self).__init__()
        self.max_action_values = max_action_values
        self.dense1 = layers.Dense(400, activation="relu")
        self.dense2 = layers.Dense(300, activation="relu")
        self.out = layers.Dense(num_action_feats, activation="tanh")

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        action = self.out(x) * self.max_action_values
        return action


class DDPG(object):
    def __init__(
        self,
        env_name,
        num_state_feats,
        num_action_feats,
        min_action_values,
        max_action_values,
        actor_lr=1e-3,
        critic_lr=1e-3,
        buffer_size=500000,
        discount=0.99,
    ):
        self.num_state_feats = num_state_feats
        self.num_action_feats = num_action_feats
        self.min_action_values = min_action_values
        self.max_action_values = max_action_values

        self.discount = discount
        self.buffer = ReplayBuffer(buffer_size)

        if env_name == "CarRacing-v0":
            self.actor = ActorNetworkConv(
                num_action_feats, min_action_values, max_action_values
            )
            self.actor_target = ActorNetworkConv(
                num_action_feats, min_action_values, max_action_values
            )
            self.critic = CriticNetworkConv()
            self.critic_target = CriticNetworkConv()
        else:
            self.actor = ActorNetwork(
                num_state_feats, num_action_feats, max_action_values
            )
            self.actor_target = ActorNetwork(
                num_state_feats, num_action_feats, max_action_values
            )
            self.critic = CriticNetwork(num_state_feats, num_action_feats)
            self.critic_target = CriticNetwork(
                num_state_feats, num_action_feats
            )

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=critic_lr
        )
        self.loss = tf.keras.losses.Huber()

    def take_exploration_action(self, state, noise_scale=0.1):
        """Takes action using self.actor network and adding noise for
        exploration."""
        act = self.actor(state) + tf.random.normal(
            shape=(self.num_action_feats,), stddev=noise_scale
        )
        return tf.clip_by_value(
            act, self.min_action_values, self.max_action_values
        )[0]

    def update_target_network(self, source_weights, target_weights, tau=0.001):
        """Updates target networks using Polyak averaging."""
        for source_weight, target_weight in zip(
            source_weights, target_weights
        ):
            target_weight.assign(
                tau * source_weight + (1.0 - tau) * target_weight
            )

    @tf.function
    def train_step(
        self,
        batch_states,
        batch_actions,
        batch_rewards,
        batch_next_states,
        batch_dones,
    ):
        # Calculate critic target.
        next_actions = self.actor_target(batch_next_states)
        next_qs = self.critic_target((batch_next_states, next_actions))
        next_qs = tf.reshape(next_qs, (-1,))
        target = batch_rewards + (1.0 - batch_dones) * self.discount * next_qs
        with tf.GradientTape() as tape:
            # Calculate critic loss.
            qs = self.critic((batch_states, batch_actions))
            qs = tf.reshape(qs, (-1,))
            critic_loss = self.loss(target, qs)

        # Calculate and apply critic gradients.
        critic_gradients = tape.gradient(
            critic_loss, self.critic.trainable_variables
        )
        self.critic_optimizer.apply_gradients(
            zip(critic_gradients, self.critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            # Calculate actor loss.
            policy_actions = self.actor(batch_states)
            policy_qs = self.critic((batch_states, policy_actions))
            actor_loss = -tf.reduce_mean(policy_qs)

        # Calculate and apply actor gradients.
        actor_gradients = tape.gradient(
            actor_loss, self.actor.trainable_variables
        )
        self.actor_optimizer.apply_gradients(
            zip(actor_gradients, self.actor.trainable_variables)
        )

        self.update_target_network(
            self.critic.weights, self.critic_target.weights, tau=0.001
        )
        self.update_target_network(
            self.actor.weights, self.actor_target.weights, tau=0.001
        )
        return actor_loss, critic_loss

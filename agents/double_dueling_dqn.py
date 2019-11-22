import tensorflow as tf
import random

from envs import ATARI_ENVS

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


class QNetworkConv(tf.keras.Model):
    """Convolutional neural network for the Atari games."""

    def __init__(self, num_actions):
        """Initializes the neural network."""
        super(QNetworkConv, self).__init__()
        self.conv1 = layers.Conv2D(32, 8, strides=(4, 4), activation="relu")
        self.conv2 = layers.Conv2D(64, 4, strides=(2, 2), activation="relu")
        self.conv3 = layers.Conv2D(64, 3, strides=(1, 1), activation="relu")
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation="relu")
        self.V = layers.Dense(1)
        self.A = layers.Dense(num_actions)
        self.Q = layers.Dense(num_actions)

    def call(self, states):
        """Forward pass of the neural network with some inputs.
        Args:
            states: tf.Tensor, batch of states.
        Returns:
            Q: tf.Tensor, the q-values of the given state for all possible
                actions.
        """
        states = tf.reshape(states, (-1, 84, 84, 1))
        x = self.conv1(states)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        V = self.V(x)
        A = self.A(x)
        Q = V + tf.subtract(A, tf.reduce_mean(A, axis=1, keepdims=True))
        return Q


class QNetwork(tf.keras.Model):
    """Dense neural network for simple games."""

    def __init__(self, num_actions):
        """Initializes the neural network."""
        super(QNetwork, self).__init__()
        self.dense1 = layers.Dense(32, activation="relu")
        self.dense2 = layers.Dense(32, activation="relu")
        self.V = layers.Dense(1)
        self.A = layers.Dense(num_actions)
        self.Q = layers.Dense(num_actions)

    def call(self, states):
        """Forward pass of the neural network with some inputs.
        Args:
            states: tf.Tensor, batch of states.
        Returns:
            Q: tf.Tensor, the q-values of the given state for all possible
                actions.
        """
        x = self.dense1(states)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)
        Q = V + tf.subtract(A, tf.reduce_mean(A, axis=1, keepdims=True))
        return Q


class DoubleDuelingDQN(object):
    """Implement the Double DQN algorithm and some helper methods."""

    def __init__(
        self,
        env_name,
        num_state_feats,
        num_actions,
        lr=1e-3,
        buffer_size=500000,
        discount=0.99,
    ):
        self.num_state_feats = num_state_feats
        self.num_actions = num_actions
        self.discount = discount
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
        result = tf.random.uniform((1,))
        if result < epsilon:
            return env.action_space.sample()
        else:
            qs = self.run_main_nn(state)
            return tf.argmax(qs[0]).numpy()  # Greedy action for state

    def update_target_network(self, source_weights, target_weights, tau=0.005):
        """Updates target network copying the weights from the source to the
        target.
        Args:
            source_weights: list, a list of the NN weights resulting of running
                nn.weights on keras. Weights to be copied from.
            target_weights: list, a list of the NN weights resulting of running
                nn.weights on keras. Weights to be copied to.
            tau: float, value in range [0, 1] for the polyak averaging value.
        """
        for source_weight, target_weight in zip(
            source_weights, target_weights
        ):
            target_weight.assign(
                tau * source_weight + (1.0 - tau) * target_weight
            )

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """Perform a training iteration on a batch of data sampled from the experience
        replay buffer.
        Returns:
            loss: float, the loss value of the current training step.
        """
        # Calculate targets.
        next_qs_main = self.main_nn(next_states)
        next_qs_argmax = tf.argmax(next_qs_main, axis=-1)
        next_action_mask = tf.one_hot(next_qs_argmax, self.num_actions)
        next_qs_target = self.target_nn(next_states)
        masked_next_qs = tf.reduce_sum(
            next_action_mask * next_qs_target, axis=-1
        )
        target = rewards + (1.0 - dones) * self.discount * masked_next_qs
        with tf.GradientTape() as tape:
            qs = self.main_nn(states)
            action_masks = tf.one_hot(actions, self.num_actions)
            masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
            loss = self.loss(target, masked_qs)
        grads = tape.gradient(loss, self.main_nn.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.main_nn.trainable_variables)
        )

        self.update_target_network(
            self.main_nn.weights, self.target_nn.weights, tau=0.005
        )
        return (loss,)

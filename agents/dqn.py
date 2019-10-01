import tensorflow as tf
import random

layers = tf.keras.layers

class ReplayBuffer(object):
    def __init__(self, size):
        self.buffer = []
        self._size = size

    def __len__(self):
        return len(self.buffer)

    def add_to_buffer(self, state, action, reward, next_state, done):
        """Adds data to experience replay buffer."""
        if len(self.buffer) == self._size:
            self.buffer = self.buffer[1:]
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, num_samples):
        batch = random.sample(self.buffer, num_samples)
        states, actions, rewards, next_states, dones = list(zip(*batch))
        return states, actions, rewards, next_states, dones


class QNetworkConv(tf.keras.Model):
    def __init__(self, num_actions):
        super(QNetworkConv, self).__init__()
        self.conv1 = layers.Conv2D(32, 3, strides=(2, 2), activation='relu') # out 48,48,32
        self.conv2 = layers.Conv2D(32, 4, strides=(2, 2), activation='relu') # out 24,24,32
        self.conv3 = layers.Conv2D(64, 4, strides=(2, 2), activation='relu') # out 12,12,64
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.out = layers.Dense(num_actions)

    def call(self, states):
        """Calls the neural network with some inputs.
        Args:
            inputs: batch of states.
        Returns:
            qs: the q-values of the given state for all possible actions.
        """
        x = self.conv1(states)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        qs = self.out(x)
        return qs


class QNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.dense1 = layers.Dense(32, activation="relu")
        self.dense2 = layers.Dense(32, activation="relu")
        self.out = layers.Dense(num_actions)

    def call(self, states):
        x = self.dense1(states)
        x = self.dense2(x)
        qs = self.out(x)
        return qs


class DQN(object):
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
        self.step = 0

        if env_name == 'CarRacing-v0':
            self.main_nn = QNetworkConv(num_actions)
            self.target_nn = QNetworkConv(num_actions)
        else:
            self.main_nn = QNetwork(num_actions)
            self.target_nn = QNetwork(num_actions)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss = tf.keras.losses.MeanSquaredError()

    def take_epsilon_greedy_action(self, state, epsilon):
        """Take random action with probability epsilon, else take best action."""
        result = tf.random.uniform((1,))
        if result < epsilon:
            return tf.random.categorical([[1. / self.num_actions]], 1)[0, 0]
        else:
            return tf.argmax(self.main_nn(state)[0]) # Greedy action for state

    def update_target_network(self, source_weights, target_weights):
        """Updates target network copying the weights from the source."""
        for source_weight, target_weight in zip(source_weights, target_weights):
            target_weight.assign(source_weight)

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """Perform a training iteration on a batch of data sampled from the experience
        replay buffer."""
        # Calculate targets.
        next_qs = self.target_nn(next_states)
        max_next_qs = tf.reduce_max(next_qs, axis=-1)
        target = rewards + (1. - dones) * self.discount * max_next_qs
        with tf.GradientTape() as tape:
            qs = self.main_nn(states)
            action_masks = tf.one_hot(actions, self.num_actions)
            masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
            loss = self.loss(target, masked_qs)
        grads = tape.gradient(loss, self.main_nn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.main_nn.trainable_variables))

        self.step += 1
        print(self.step)
        if self.step % 100 == 0:
            self.update_target_network(self.main_nn.weights, self.target_nn.weights)
        return loss

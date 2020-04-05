import tensorflow_probability as tfp
import tensorflow as tf

layers = tf.keras.layers


class PolicyNet(tf.keras.Model):
    def __init__(self, num_actions):
        super(PolicyNet, self).__init__()
        self.dense1 = layers.Dense(200, activation='relu')
        self.dense2 = layers.Dense(200, activation='relu')
        self.mean = layers.Dense(num_actions, activation=None)
        self.log_std = layers.Dense(num_actions, activation=None)
  
    @tf.function
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = tf.exp(log_std)
        pi = mean + tf.random.normal(tf.shape(mean), dtype=tf.float32) * std
        action = tf.math.tanh(pi) # Convert to range (-1, 1).
        log_prob = tfp.distributions.Normal(loc=mean, scale=std).log_prob(pi)
        log_prob -= tf.math.log(1 - tf.math.pow(action, 2) + 1e-6)
        return pi, action, log_prob, mean, log_std

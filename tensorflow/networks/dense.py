import tensorflow as tf


layers = tf.keras.layers


class Dense(tf.keras.Model):
    """Dense neural network for simple games."""
    __name__ = "Dense"

    def __init__(self, num_actions):
        """Initializes the neural network."""
        super(Dense, self).__init__()
        self.dense1 = layers.Dense(32, activation="relu")
        self.dense2 = layers.Dense(32, activation="relu")
        self.out = layers.Dense(num_actions)

    @tf.function
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

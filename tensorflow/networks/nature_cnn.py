import tensorflow as tf

layers = tf.keras.layers


class NatureCNN(tf.keras.Model):
    """Convolutional neural network for the Atari games."""

    def __init__(self, num_actions):
        """Initializes the neural network."""
        super(NatureCNN, self).__init__()
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
        # states = tf.cast(states, tf.float32)
        x = self.conv1(states)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.out(x)

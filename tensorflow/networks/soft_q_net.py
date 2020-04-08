import tensorflow as tf

layers = tf.keras.layers


class SoftQNet(tf.keras.Model):
    def __init__(self):
        super(SoftQNet, self).__init__()
        self.dense1 = tf.keras.layers.Dense(
            200,
            activation="relu",
            kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
            bias_initializer=tf.keras.initializers.Zeros(),
        )
        self.dense2 = tf.keras.layers.Dense(
            200,
            activation="relu",
            kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
            bias_initializer=tf.keras.initializers.Zeros(),
        )
        self.out = tf.keras.layers.Dense(1, activation=None)

    @tf.function
    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.out(x)
        return out

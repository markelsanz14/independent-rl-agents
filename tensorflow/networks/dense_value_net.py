import tensorflow as tf

layers = tf.keras.layers


class ValueNet(tf.keras.Model):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.dense1 = tf.keras.layers.Dense(200, activation='relu')
        self.dense2 = tf.keras.layers.Dense(200, activation='relu')
        self.out = tf.keras.layers.Dense(1, activation=None)

    @tf.function
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        out = self.out(x)
        return out

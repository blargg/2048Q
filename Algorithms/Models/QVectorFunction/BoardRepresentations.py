import tensorflow as tf


class OneHotBoard:
    def __init__(self, stateDims=[4, 4], depth=12):
        self.input_state = tf.placeholder(tf.int32, shape=[None] + stateDims)
        self.encoded = tf.one_hot(self.input_state, depth)
        self.state_representation = tf.layers.flatten(self.encoded)

import tensorflow as tf


class ParabolicModel:
    def __init__(self):
        """Simple parabolic model.
        Tensors:
          x: input, placeholder for the input of the function
          prediction: output, the prediction based on the learned function
          actual_value: input, the actual y used for training
          train_op:
        """
        # variables to change the learning rate as time goes on
        count = tf.get_variable("train_count", shape=[], dtype=tf.int32,
                                initializer=tf.initializers.zeros)
        # learning rate bottoms out
        learning_rate = tf.maximum(10 / (count + 100), 0.0000001)

        self.x = tf.placeholder(tf.float32, [], "x")
        self.actual_value = tf.placeholder(tf.float32, [], "actual_value")
        self.w = tf.get_variable("w", shape=[3],
                                 initializer=tf.initializers.random_uniform())
        one = tf.ones_like(self.x)
        polynomial_x = tf.stack([tf.pow(self.x, 2), self.x, one])
        self.prediction = tf.reduce_sum(polynomial_x * self.w,
                                        name='prediction')
        loss = tf.nn.l2_loss(self.actual_value - self.prediction)

        opt = tf.train.AdamOptimizer(learning_rate)
        # opt = tf.train.GradientDescentOptimizer(learning_rate)
        self.train = opt.minimize(loss, global_step=count)


def examplefun(x):
    return 3 * x * x + 0.5 * x - 2


def train(sess, model):
    for x in range(-100, 100):
        sess.run(model.train,
                 {model.x: x, model.actual_value: examplefun(x)})

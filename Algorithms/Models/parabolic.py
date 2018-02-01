import tensorflow as tf
import random


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
        init_lr = tf.initializers.constant(0.001)
        self.learning_rate = tf.get_variable("learning_rate",
                                             shape=[],
                                             trainable=False,
                                             initializer=init_lr,
                                             dtype=tf.float32)

        self.x = tf.placeholder(tf.float32, [], "x")
        self.actual_value = tf.placeholder(tf.float32, [], "actual_value")
        self.w = tf.get_variable("w", shape=[3],
                                 initializer=tf.initializers.random_uniform(),
                                 dtype=tf.float32, )
        one = tf.ones_like(self.x)
        polynomial_x = tf.stack([tf.pow(self.x, 2), self.x, one])
        self.prediction = tf.reduce_sum(polynomial_x * self.w,
                                        name='prediction')
        l2_loss = tf.nn.l2_loss(self.actual_value - self.prediction)
        self.loss = tf.log(l2_loss + 1e-6)  # having trouble with large losses

        opt = tf.train.AdamOptimizer(self.learning_rate)
        self.train = opt.minimize(self.loss, global_step=count)


def examplefun(x):
    return 3 * x * x + 0.5 * x - 2


data = [(x, examplefun(x)) for x in range(-100, 100)]


def train(sess, model):
    random.shuffle(data)
    for x, y in data:
        sess.run(model.train,
                 {model.x: x, model.actual_value: y})

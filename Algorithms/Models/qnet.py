import Algorithms.QLearning as q
import tensorflow as tf
import numpy as np


class QNet(q.ReinforcementLearner):
    """Q-learning algorithm that models the that accpets a QFunction.
    This algorithm can only handle a small, finite action space
    (less than 10 should be fine)

    Q function accepts an array with shape stateDims and returns a vector
    of size numActions"""

    def __init__(self,
                 makeFunction,
                 learning_rate=0.001,
                 discount=0.9):
        """Constructs a tensorflow graph on the default graph."""
        # variables constant through initialization

        # settings
        self.discount = discount
        self.learning_rate = learning_rate
        self.learning_rate_tf =\
            tf.get_variable("learning_rate",
                            initializer=tf.constant(learning_rate))

        # Q-function construction
        self.q_function = makeFunction(trainable=True)
        self.state = self.q_function.state
        self.actions = self.q_function.actions

        # placholder for target action vector
        self.target_actions = tf.placeholder(self.actions.dtype,
                                             shape=self.actions.get_shape())

        actions_shape = self.actions.get_shape()
        self.numActions = int(actions_shape[1])

        # training construction
        diff_sq = tf.square(self.actions - self.target_actions)
        self.loss = tf.reduce_mean(diff_sq, name="loss")
        optimizer = tf.train.AdamOptimizer(self.learning_rate_tf)
        self.train = optimizer.minimize(self.loss)
        # create a session to use in this class
        self.sess = tf.Session()

    def initializeVariables(self):
        self.sess.run(tf.global_variables_initializer())

    def close(self):
        self.sess.close()

    def _setGDLearningRate(self, learning_rate):
        self.sess.run(self.learning_rate_tf.assign(learning_rate))

    def actionVectorBatch(self, state):
        """Returns a vector of Q-vaules for each state. Each value in the
        vector is the q-value for the action of that index

        state: a list of states. Dimentions should be self.state.get_shape()
        """
        return self.sess.run(self.actions, {self.state: state})

    def actionVector(self, state):
        """Single state variant of actionVectorBatch"""
        return self.actionVectorBatch([state])[0]

    def chooseActionBatch(self, state):
        """Evaluates the model and selects the action with the highest Q-value

        state: a list of states. Dimentions should be self.state.get_shape()
        """
        action_vec = self.actionVectorBatch(state)
        return np.argmax(action_vec, 1)

    def chooseAction(self, state):
        """Single state version of chooseActionBatch"""
        return self.chooseActionBatch([state])[0]

    def save(self, filename):
        """saves the session to the filename"""
        saver = tf.train.Saver()
        saver.save(self.sess, filename)

    def load(self, filename):
        """loads the session from the given file
        should be the same model as was saved"""
        saver = tf.train.Saver()
        saver.restore(self.sess, filename)

    def observeResult(self, state, action, nextState, reward):
        assert action >= 0 and action < self.numActions,\
            "action = {}, but must be from 0 to {}"\
            .format(action, self.numActions)

        # run updates based on example
        predicted = self.actionVector(state)
        q_next = max(self.actionVector(nextState))
        actual = predicted.copy()
        # TODO: using learning rate in the conventional way should not be
        # needed for neural nets. It can instead be the learning rate of the
        # gradient descent step.
        actual[action] =\
            (1 - self.learning_rate) * actual[action] +\
            self.learning_rate * (reward + self.discount * q_next)

        self.sess.run(self.train,
                      {self.state: [state], self.target_actions: [actual]})

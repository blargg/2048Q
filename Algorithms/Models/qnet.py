import Algorithms.QLearning as q
import tensorflow as tf
import numpy as np
import random
import math
from util.random import bernoulli
from util.fifo import Fifo


class QNet(q.ReinforcementLearner):
    """Q-learning algorithm that models the that accpets a QFunction.
    This algorithm can only handle a small, finite action space
    (less than 10 should be fine)

    Q function accepts an array with shape stateDims and returns a vector
    of size numActions"""

    def __init__(self,
                 makeFunction,
                 learning_rate=0.001,
                 discount=0.9,
                 epsilon=0.0,
                 possibleActions=None,
                 memorySize=200):
        """Constructs a tensorflow graph on the default graph.

        Args:
          makeFunction: constructs a tensorflow graph to use
                        must have input tensor state
                        output tensor actions
          learning_rate: controls how fast the model will change based on
                         new observations
          discount: High discount factor will cause learning to concider future
                    actions more heavily. 1 will concider any future reward
                    to be just as important as the next reward
          epsilon: how often to explore a random action
          possibleActions: function that accepts a state and returns a vector
                           of possible actions (True for possible)
          memorySize: The number of recent observations to remember"""
        # variables constant through initialization

        # settings
        self.discount = discount
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.possibleActions = possibleActions
        self.learning_rate_tf =\
            tf.get_variable("learning_rate",
                            initializer=tf.constant(learning_rate))

        # initialize memory
        self.recentObservations = Fifo(memorySize)

        # Q-function construction
        with tf.variable_scope("QFunction"):
            self.q_function = makeFunction(trainable=True)
        self.state = self.q_function.state
        self.actions = self.q_function.actions

        # Reference Q-function
        with tf.variable_scope("QReference"):
            self.q_reference = makeFunction(trainable=False)

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

    def copyQFuncToReference(self):
        vars = self.q_function.get_all_variables()
        refs = self.q_reference.get_all_variables()
        self.sess.run([ref.assign(var) for (var, ref) in zip(vars, refs)])

    def setKeepProb(self, prob):
        self.sess.run(self.q_function.keep_prob.assign(prob))

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

    def actionsAndRefBatch(self, startStates, nextStates):
        tensors = (self.actions, self.q_reference.actions)
        inputs = {self.state: startStates,
                  self.q_reference.state: nextStates}
        return self.sess.run(tensors, inputs)

    def actionAndRef(self, startState, nextState):
        startStates = [startState]
        nextStates = [nextState]
        (actions, refs) = self.actionsAndRefBatch(startStates, nextStates)
        return (actions[0], refs[0])

    def chooseActionBatch(self, state):
        """Evaluates the model and selects the action with the highest Q-value

        state: a list of states. Dimentions should be self.state.get_shape()
        """
        action_vec = self.actionVectorBatch(state)

        action_vec = self.maskActions(action_vec, state)
        result = np.argmax(action_vec, 1)
        if self.epsilon > 0.000001:
            # TODO, should obey maskActions
            mapExplore = np.vectorize(self.__exploreAction)
            return mapExplore(result)
        return result

    def maskActions(self, action_vec, states):
        """If an action is not possible, based on our model knowledge, map it's
        Q-value to Negative Infinity"""
        if self.possibleActions is None:
            return action_vec

        def Qvalue(isPossible):
            if isPossible:
                return 0.0
            else:
                return -1 * math.inf
        possible = np.array([self.possibleActions(state) for state in states])
        mapQVal = np.vectorize(Qvalue)
        return action_vec + mapQVal(possible)

    def __exploreAction(self, action):
        if bernoulli(self.epsilon) == 1:
            return random.randrange(self.numActions)
        else:
            return action

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
        obs = q.Observation(state, action, nextState, reward)
        self.recentObservations.add(obs)
        if self.recentObservations.nextAddCycles():
            self.trainRecent()

    def trainRecent(self):
        self.copyQFuncToReference()
        indecies = [i for i in range(self.recentObservations.size())]
        random.shuffle(indecies)
        for i in indecies:
            # TODO only get populated indecies
            observation = self.recentObservations.getElement(i)
            self.trainObservation(observation)

    def trainObservation(self, observation):
        self.trainStep(observation.startState,
                       observation.action,
                       observation.nextState,
                       observation.reward)

    def trainStep(self, state, action, nextState, reward):
        assert action >= 0 and action < self.numActions,\
            "action = {}, but must be from 0 to {}"\
            .format(action, self.numActions)

        # TODO make a batch version
        # run updates based on example

        (actionVals, nextActionVals) = self.actionAndRef(state, nextState)
        q_next = max(nextActionVals)
        actual = actionVals.copy()
        # TODO: using learning rate in the conventional way should not be
        # needed for neural nets. It can instead be the learning rate of the
        # gradient descent step.
        actual[action] =\
            (1 - self.learning_rate) * actual[action] +\
            self.learning_rate * (reward + self.discount * q_next)

        self.sess.run(self.train,
                      {self.state: [state], self.target_actions: [actual]})

from Algorithms.QLearning import ReinforcementLearner
from util.random import bernoulli
import random


class TableLearner(ReinforcementLearner):
    """Q-learning algorithm that records the Q function with a python
    dictionary. Stores the Q function value of each state X action in the
    table"""
    # initial value all outputs of the Q function
    initialQ = 0.0
    # probability of taking a random action
    epsilon = 0.1

    def __init__(self, possibleActions,
                 learningRate=0.9, discount=0.5):
        self.qTable = {}
        self.possibleActions = possibleActions
        self.learningRate = learningRate
        self.discount = discount

    def q(self, state, action):
        return self.qTable.get((state, action), self.initialQ)

    def chooseAction(self, state):
        actions = self.possibleActions(state)
        if bernoulli(self.epsilon) == 1:
            return random.choice(actions)
        else:
            return max(actions, key=lambda action: self.q(state, action))

    def observeResult(self, state, action, nextState, reward):
        """Update state X action Q value based on the reward observed and
        the next observed state"""
        oldQ = self.q(state, action)
        # nextQ is the predicted Q value for taking the optimal action in
        # nextState
        nextQ = max([self.q(nextState, nextAction)
                    for nextAction in self.possibleActions(nextState)])
        learnedQ = reward + self.discount * nextQ
        updatedQ = (1 - self.learningRate) * oldQ +\
            self.learningRate * learnedQ
        self.qTable[(state, action)] = updatedQ

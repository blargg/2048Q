import Algorithms.QLearning as q
import random


class RandomIndexAction(q.ReinforcementLearner):
    """Choses a random action"""

    def __init__(self, maxIndex):
        self.maxIndex = maxIndex

    def chooseAction(self, state):
        return random.randrange(self.maxIndex)

    def observeResult(self, state, action, nextState, reward):
        """Does nothing"""
        pass

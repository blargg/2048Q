import Algorithms.QLearning as q


class CyclicIndexAction(q.ReinforcementLearner):
    """Choses a random action"""

    def __init__(self, numActions):
        self.lastAction = 0
        self.numActions = numActions

    def chooseAction(self, state):
        self.lastAction = (self.lastAction + 1) % self.numActions
        return self.lastAction

    def observeResult(self, state, action, nextState, reward):
        """Does nothing"""
        pass

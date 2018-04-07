import Algorithms.QLearning as q


class FixedAction(q.ReinforcementLearner):
    """Choses a random action"""

    def __init__(self, action):
        self.action = action

    def chooseAction(self, state):
        return self.action

    def observeResult(self, state, action, nextState, reward):
        """Does nothing"""
        pass

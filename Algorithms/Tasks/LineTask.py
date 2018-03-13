from Algorithms.QLearning import ReinforcementTask
import random


class LineState:
    """State of the LineTask. Simple wrapper around int"""
    def __init__(self, val):
        self.value = val

    def set(self, val):
        self.value = val

    def copy(self):
        return LineState(self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        return "<Line State " + str(self.value) + ">"


# TODO it's pretty unclear that we have to call from the class, rather than
# an instanciated object
class LineTask(ReinforcementTask):
    """Simple task, where states are the numbers on the number line, actions
    are numbers to jump to. Higher rewards for getting close to 0"""
    def transition(state, action):
        state.set(action)

    def reward(state, action):
        baseReward = 1 / (1 + abs(action))
        # noise = random.normalvariate(0, 0.01)
        noise = 0
        return baseReward + noise

    def startState():
        return LineState(random.choice(range(-10, 10)))

    def actions(state):
        location = state.value
        return [action for action in range(location - 1, location + 2)]

    def isEndState(state):
        if state.value < -9:
            return True
        if state.value > 9:
            return True
        if state.value == 0:
            return True

        return False

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


class LineTask(ReinforcementTask):
    """Simple task, where states are the numbers on the number line, actions
    are numbers to jump to. Higher rewards for getting close to 0"""
    def transition(state, action):
        state.set(action)

    def reward(state, action):
        baseReward = 1 / (1 + abs(action))
        noise = random.normalvariate(0, 0.1)
        return baseReward + noise

    def startState():
        return LineState(random.choice(range(-10, 10)))

    def actions(state):
        location = state.value
        return [action for action in range(location - 1, location + 2)]

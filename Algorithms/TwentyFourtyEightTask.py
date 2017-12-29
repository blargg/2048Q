import Algorithms.QLearning as Q
from Game.Model import newBoard, act, shiftBoard


class TwentyFourtyEightTask(Q.ReinforcementTask):
    """Task to play the game 2048.
    See Game.Model for more information"""
    def transition(state, action):
        """Slide the tiles and add random piece"""
        return act(state, action)

    def reward(state, action):
        """Reward is based on the difference in the highest value on the
        board before and after the pieces have shifted"""
        nextState = state.copy()
        shiftBoard(nextState, action)
        return nextState.max() - state.max()

    def startState():
        """Start the game with a new board"""
        return newBoard()

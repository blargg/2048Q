import Algorithms.QLearning as Q
import Game.Model as M
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

    def isEndState(state):
        """Game ends when no more moves are possible"""
        return M.isGameOver(state)


class IndexedTask(Q.ReinforcementTask):
    """Same as TwentyFortyEightTask, but accepts an indexed number 0 to 3 as
    an action"""
    def transition(state, action):
        actionEnum = M.fromIndex(action)
        return TwentyFourtyEightTask.transition(state, actionEnum)

    def reward(state, action):
        actionEnum = M.fromIndex(action)
        return TwentyFourtyEightTask.reward(state, actionEnum)

    def startState():
        return TwentyFourtyEightTask.startState()

    def isEndState(state):
        return TwentyFourtyEightTask.isEndState(state)

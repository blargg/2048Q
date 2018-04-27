import Algorithms.QLearning as Q
import Game.Model as M
from Game.Model import newBoard, act


class TwentyFourtyEightTask(Q.ReinforcementTask):
    """Task to play the game 2048.
    See Game.Model for more information"""
    def transition(state, action):
        """Slide the tiles and add random piece"""
        return act(state, action)

    def startState():
        """Start the game with a new board"""
        return newBoard()

    def isEndState(state):
        """Game ends when no more moves are possible"""
        return M.isGameOver(state)

    def stateEq(state1, state2):
        return (state1 == state2).all()


class IndexedTask(TwentyFourtyEightTask):
    """Same as TwentyFortyEightTask, but accepts an indexed number 0 to 3 as
    an action"""
    def transition(state, action):
        actionEnum = M.fromIndex(action)
        return TwentyFourtyEightTask.transition(state, actionEnum)

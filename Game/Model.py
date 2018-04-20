from enum import Enum, unique
import numpy as np
import random
import math


@unique
class Action(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


def toIndex(action):
    lookup = {Action.UP: 0,
              Action.DOWN: 1,
              Action.LEFT: 2,
              Action.RIGHT: 3}
    return lookup.get(action)
    pass


def fromIndex(index):
    lookup = {0: Action.UP,
              1: Action.DOWN,
              2: Action.LEFT,
              3: Action.RIGHT}
    return lookup.get(index)
    pass


def makeBoard(rows, columns, startVal):
    return np.array([[startVal for x in range(columns)] for y in range(rows)])


def emptyBoard():
    return makeBoard(4, 4, 0)


def newBoard():
    board = emptyBoard()
    addRandomBlock(board)
    return board


def randomBoard(smallest=0, largest=10):
    vals = [math.floor(random.uniform(smallest, largest+1)) for _ in range(16)]
    return np.reshape(vals, (4, 4))


def printBoard(board):
    for row in board:
        print(row)
    return


def shiftValues(row):
    """Shifts values over to the left, merging equal values.
    Merged values increment by one from the pre-merge values
    Assumes all values are numeric.
    """
    assert all([elem >= 0 for elem in row]),\
        "all the elements must be greater than one"
    nextRow = []
    lastValue = None
    for val in row:
        if val == 0:
            pass
        elif lastValue is None:
            lastValue = val
        elif lastValue == val:
            lastValue = None
            nextRow.append(val+1)
        else:
            nextRow.append(lastValue)
            lastValue = val
    if lastValue is not None:
        nextRow.append(lastValue)
    lendiff = len(row) - len(nextRow)
    nextRow.extend([0 for x in range(lendiff)])
    return nextRow


def shiftBoard(board, action):
    # assume that the board is square
    # can adjust to support rectangular boards or irregular size rows
    assert isinstance(board, np.ndarray), \
        "assume these are numpy arrays"
    shape = board.shape

    if action == Action.UP:
        for i in range(shape[1]):
            board[:, i] = shiftValues(board[:, i])
    elif action == Action.DOWN:
        for i in range(shape[1]):
            board[::-1, i] = shiftValues(board[::-1, i])
    elif action == Action.RIGHT:
        for i in range(shape[0]):
            board[i, ::-1] = shiftValues(board[i, ::-1])
    else:
        for i in range(shape[0]):
            board[i, :] = shiftValues(board[i, :])


def randomBlock():
    return random.choice([1, 2])


def addRandomBlock(board):
    zeroIndecies = np.transpose(np.where(board == 0))
    if zeroIndecies.shape[0] == 0:
        return
    index = random.choice(zeroIndecies)
    board[index[0], index[1]] = randomBlock()


def act(board, action):
    shiftBoard(board, action)
    addRandomBlock(board)


def possibleAction(board, action):
    """Checks if the action will continue the game.
    An action does not continue the game if the shiftBoard action will not
    create an empty space to place a new tile"""
    # TODO this can be more efficient
    # as soon as we collapse 2 squares, or find a 0, we know it's possible
    shift = board.copy()
    shiftBoard(shift, action)
    if np.any(shift == 0):
        return True
    return False


def possibleActionVector(board):
    """Returns a array of possible actions. The index in the array is the
    action, the value at that index is 1 if the action is possible,
    or 0 otherwise"""
    actions = [False for x in range(len(Action))]

    for a in Action:
        if possibleAction(board, a):
            actions[toIndex(a)] = True
    return actions


def isGameOver(board):
    """Game ends when there are no possible actions.
    See possibleAction."""

    # TODO would like to make an iterable for this and possibleActionVector
    for a in Action:
        if possibleAction(board, a):
            return False
    return True


def highestTile(board):
    return board.max()

from enum import Enum
import numpy as np
import random


class Action(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


def makeBoard(rows, columns, startVal):
    return np.array([[startVal for x in range(columns)] for y in range(rows)])


def emptyBoard():
    return makeBoard(4, 4, 0)


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


def isGameOver(board):
    """Game ends when there is no way to shift the board to make empty
    spaces"""
    left = shiftBoard(board.copy(), Action.LEFT)
    if np.any(left == 0):
        return False

    right = shiftBoard(board.copy(), Action.RIGHT)
    if np.any(right == 0):
        return False

    up = shiftBoard(board.copy(), Action.UP)
    if np.any(up == 0):
        return False

    down = shiftBoard(board.copy(), Action.DOWN)
    if np.any(down == 0):
        return False

    return True

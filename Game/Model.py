from enum import Enum
import numpy as np


class Action(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


def makeBoard(rows, columns, startVal):
    return [[startVal for x in range(columns)] for y in range(rows)]


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
    nextRow = []
    lastValue = 0
    for val in row:
        if lastValue == 0:
            lastValue = val
        elif lastValue == val:
            lastValue = 0
            nextRow.append(val+1)
        else:
            nextRow.append(lastValue)
            lastValue = val
    nextRow.append(lastValue)
    lendiff = len(row) - len(nextRow)
    nextRow.extend([0 for x in range(lendiff)])
    return nextRow


def act(board, action):
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

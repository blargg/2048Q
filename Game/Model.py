from enum import Enum


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


def act(board, action):
    pass

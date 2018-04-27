from hypothesis import given
import hypothesis.strategies as st

import numpy as np

from Game.Model import shiftValues, isGameOver, shiftBoard, Action
import Game.Model as g


def listOfSize(elements, size):
    return st.lists(elements, min_size=size, max_size=size)


# sampling strategies
actions = st.sampled_from([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT])


def boards(minVal=0, maxVal=15):
    lists = listOfSize(st.integers(minVal, maxVal), 16)

    def shapeToBoard(elems):
        return np.reshape(elems, (4, 4))

    return lists.map(shapeToBoard)


@given(st.integers(0, 3))
def test_index_iso_to_action(index):
    assert index == g.toIndex(g.fromIndex(index)),\
        "toIndex and fromIndex should be isomorphic"


@given(st.lists(st.integers(0, 100)))
def test_shift_ends_with_zeros(xs):
    """Tests that after the list is shifted,
    the first zero is only followed by zeros"""
    (lst, _) = shiftValues(xs)
    afterZero = False
    for x in lst:
        if x == 0:
            afterZero = True
        if afterZero:
            assert x == 0, "everything after the first zero should be zero"


@given(st.lists(st.integers(0, 100)))
def test_shift_perserves_sq_sum(xs):

    def sq_sum(list):
        # need to filter out 0, because they represent empty sq, rather than 0
        ls = filter(lambda x: x > 0, list)
        return sum([2 ** l for l in ls])

    original_sum = sq_sum(xs)
    (shifted, _) = shiftValues(xs)
    shifted_sum = sq_sum(shifted)
    assert original_sum == shifted_sum,\
        "Sum of squares elements should not change after shifting"


@given(st.lists(st.integers(0, 100)))
def test_always_same_length(xs):
    (lst, _) = shiftValues(xs)
    assert len(lst) == len(xs)


@given(st.lists(st.integers(0, 15)))
def test_shiftValues_nonNegative(xs):
    (_, score) = shiftValues(xs)
    assert score >= 0, \
        "Score should always be non-negative"


@given(boards(),
       actions)
def test_act_score_nonNegative(board, action):
    score = g.act(board, action)
    assert score >= 0, \
        "Score should always be non-negative"


def test_shiftValues():
    initial = [1, 1, 0, 2, 0, 2, 2, 2]
    (result, _) = shiftValues(initial)
    assert result == [2, 3, 3, 0, 0, 0, 0, 0], "Must match this example case"


def test_game_over():
    endGameBoard = np.array(range(1, 17)).reshape((4, 4))
    assert isGameOver(endGameBoard), "No way to make empty space; game over"

    endGameBoard = np.array(range(0, 16)).reshape((4, 4))
    assert not isGameOver(endGameBoard),\
        "There is still an empty space on the board, so the game continues"


def rotate_board(board):
    """Takes a list and returns array"""
    arr = np.array(board)
    assert arr.shape[0] == arr.shape[1],\
        "rotate can only be called on square arrays"
    rotated = np.flip(arr.transpose(), 1)
    return rotated


def rotate_action(action):
    if action == Action.UP:
        return Action.RIGHT
    if action == Action.DOWN:
        return Action.LEFT
    if action == Action.LEFT:
        return Action.UP
    if action == Action.RIGHT:
        return Action.DOWN
    return None


@given(actions,
       boards(0, 100))
def test_rotationalSym(action, board):
    board1 = board
    board2 = rotate_board(board1.copy())
    action1 = action
    action2 = rotate_action(action)

    shiftBoard(board1, action1)
    board1 = rotate_board(board1)

    shiftBoard(board2, action2)

    assert np.array_equal(board1, board2), "boards must equal after rotation"

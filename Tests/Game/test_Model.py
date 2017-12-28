from hypothesis import given
import hypothesis.strategies as st

import numpy as np

from Game.Model import shiftValues, isGameOver


@given(st.lists(st.integers(0, 100)))
def test_shift_ends_with_zeros(xs):
    """Tests that after the list is shifted,
    the first zero is only followed by zeros"""
    lst = shiftValues(xs)
    afterZero = False
    for x in lst:
        if x == 0:
            afterZero = True
        if afterZero:
            assert x == 0, "everything after the first zero should be zero"


@given(st.lists(st.integers(0, 100)))
def test_always_same_length(xs):
    lst = shiftValues(xs)
    assert len(lst) == len(xs)


def test_shiftValues():
    initial = [1, 1, 0, 2, 0, 2, 2, 2]
    result = shiftValues(initial)
    assert result == [2, 3, 3, 0, 0, 0, 0, 0], "Must match this example case"


def test_game_over():
    endGameBoard = np.array(range(16)).reshape((4, 4))
    assert isGameOver(endGameBoard), "No way to make empty space; game over"

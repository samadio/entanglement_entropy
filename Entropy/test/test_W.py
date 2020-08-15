from numba import njit, prange
from states import *
from unittest import TestCase
import itertools

@njit(parallel=True)
def first_column(chosens: np.array, n_rows: int, binary_combinations: np.array) -> np.array:
    results = np.zeros(n_rows)
    powers = np.power(2.0, np.flip(chosens))
    for c in prange(n_rows):
        results[c] = np.dot(powers, binary_combinations[c])
    return results


@njit(parallel=True)
def col_gaps(notchosens: np.array, n_cols: int, binary_combinations: np.array) -> np.array:
    results = np.zeros(n_cols)
    powers = np.power(2.0, np.flip(notchosens))
    for c in prange(n_cols):
        results[c] = np.dot(powers, binary_combinations[c])
    return results


def matrix_from_state_improved(state: np.ndarray, chosen: list, notchosen: list) -> np.ndarray:
    n_rows = 2 ** len(chosen)
    n_cols = 2 ** len(notchosen)
    chosen = np.sort(chosen)

    binary_combinations = np.array(list(map(list, itertools.product([0, 1], repeat=len(chosen)))), dtype=float)

    first_col_idx = np.array([first_column(np.array(chosen), n_rows, binary_combinations)])

    binary_combinations = np.array(list(map(list, itertools.product([0, 1], repeat=len(notchosen)))), dtype=float)
    gaps = col_gaps(np.array(notchosen), n_cols, binary_combinations)

    mappa = np.repeat(first_col_idx.T, 2 ** len(notchosen), axis=1)
    mappa = np.array(mappa + gaps, dtype=int)
    return state[mappa].reshape((n_rows, n_cols))


class Test(TestCase):

    def test_reduced_matrix(self):
        state = np.array([0, 1 / 2, 1 / 2, 0, 0, 1 / 2, 0, 1 / 2])
        nonzero = [1, 2, 5, 7]
        #sparse_state = bip.coo_matrix(([1 / 2, 1 / 2, 1 / 2, 1 / 2], (nonzero, [0, 0, 0, 0])), shape=(8, 1)).tocsr()
        for chosen in [[0], [1], [2]]:
            W = matrix_from_state_improved(state, chosen, bip.notchosen(chosen, 3))
            myreduced = W.dot(W.T)
            qreduced = qt.quantum_info.partial_trace(qt.quantum_info.Statevector(state), bip.notchosen(chosen, 3)).data. \
                real.astype(dtype=float, copy=False)
            np.testing.assert_array_almost_equal(myreduced, qreduced)
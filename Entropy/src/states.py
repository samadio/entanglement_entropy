from math import log2 as log2
from IQFT import *
import numpy as np
from auxiliary import auxiliary as aux, bipartitions as bip
from numba import njit, prange
import itertools


def construct_modular_state(k: int, L: int, nonzero_elements_decimal_idx: list) -> bip.coo_matrix:
    data = np.ones(2 ** k) * 2 ** (- k / 2)
    col = np.zeros(2 ** k)
    return coo_matrix((data, (nonzero_elements_decimal_idx, col)), shape=(2 ** (k + L), 1)).tocsr()


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


@njit(parallel=True)
def get_density_matrix_from_W_dense(W: np.array) -> np.array:
    return np.dot(W, np.conj(W).T)


def W_from_state_numba(state: np.ndarray, chosen: list, notchosen: list) -> np.ndarray:
    n_rows = 2 ** len(chosen)
    n_cols = 2 ** len(notchosen)
    ch = np.sort(chosen)
    nch = np.sort(notchosen)

    binary_combinations = np.array(list(map(list, itertools.product([0, 1], repeat=len(ch)))), dtype=float)
    first_col_idx = np.array([first_column(np.array(ch), n_rows, binary_combinations)]).astype(int)

    binary_combinations = np.array(list(map(list, itertools.product([0, 1], repeat=len(nch)))), dtype=float)
    gaps = col_gaps(np.array(nch), n_cols, binary_combinations).astype(int)

    mappa = np.repeat(first_col_idx.T, n_cols, axis=1)
    mappa = mappa + gaps
    return state[mappa]


def density_matrix_from_state_numba(state: np.ndarray, chosen: list, notchosen: list) -> np.ndarray:
    W = W_from_state_numba(state, chosen, notchosen)
    return get_density_matrix_from_W_dense(W)


def density_matrix_from_state_dense(state: np.ndarray, chosen: list, notchosen: list) ->np.ndarray:
    """
        Construct and return matrix W s.t. W.dot(W.T)==reduced density matrix for state after IQFT

    :param state: state of the system
    :param chosen: observable qubits
    :param notchosen: qubits to trace away
    :return:  W
    """
    return density_matrix_from_state_numba(state, chosen, notchosen)


def W_from_state_sparse(state: coo_matrix, chosen: list, notchosen: list):
    """
        Construct and return matrix W s.t. W.dot(np.conj(W).T)==reduced density matrix for modular exponentiation state

    :param state: state of the system
    :param chosen: observable qubits
    :param notchosen: qubits to trace away
    :return:  W
    """

    nonzero_idx, _ = state.nonzero()
    nonzero_idx_binary = [aux.decimal_to_binary(idx, int(log2(state.shape[0]))) for idx in nonzero_idx]
    row = [aux.to_decimal(aux.select_components(i, chosen)) for i in nonzero_idx_binary]
    col = [aux.to_decimal((aux.select_components(i, notchosen))) for i in nonzero_idx_binary]
    number_of_nonzeros = len(nonzero_idx)
    norm = number_of_nonzeros ** (- 1 / 2)
    data = np.ones(number_of_nonzeros) * norm
    return coo_matrix((data, (row, col)), shape=(2 ** len(chosen), 2 ** len(notchosen))).tocsr()

def density_matrix_from_state_sparse(state: coo_matrix, chosen: list, notchosen: list) ->coo_matrix:
    """
        Construct and return matrix W s.t. W.dot(np.conj(W).T)==reduced density matrix for state after IQFT

    :param state: state of the system
    :param chosen: observable qubits
    :param notchosen: qubits to trace away
    :return:  rho
    """
    W = W_from_state_sparse(state, chosen, notchosen)
    return W.dot(coo_matrix.conjugate(W).T)

# -----------------------------------------------------------------
# unused functions
'''
def slicing_index(i: int, L: int) -> list:
    """auxiliary function"""
    return [i % (2 ** L) + m * 2 ** L for m in range(2 ** (2 * L))]
'''
from itertools import combinations as combinations
from math import log2 as log2
from IQFT import *
import numpy as np
import scipy
from scipy.sparse import identity as sparse_identity
from auxiliary import auxiliary as aux, bipartitions as bip
from jax.numpy.linalg import eigvalsh as jeigh


def construct_modular_state(k: int, L: int, nonzero_elements_decimal_idx: list) -> bip.coo_matrix:
    data = np.ones(2 ** k) * 2 ** (- k / 2)
    col = np.zeros(2 ** k)
    return bip.coo_matrix((data, (nonzero_elements_decimal_idx, col)), shape=(2 ** (k + L), 1)).tocsr()


def matrix_from_sparse_modular_state(state: scipy.sparse.coo_matrix, chosen: list, notchosen: list,
                                     sparse: bool = True):
    """
        Construct and return matrix W s.t. W.dot(W.T)==reduced density matrix for modular exponentiation state

    :param state: state of the system
    :param chosen: observable qubits
    :param notchosen: qubits to trace away
    :param sparse: if True, W is scipy.sparse, otherwise numpy.ndarray
    :return:  W
    """
    nonzero_idx, _ = state.nonzero()
    nonzero_idx_binary = [aux.decimal_to_binary(idx, int(log2(state.shape[0]))) for idx in nonzero_idx]
    row = [aux.to_decimal(aux.select_components(i, chosen)) for i in nonzero_idx_binary]
    col = [aux.to_decimal((aux.select_components(i, notchosen))) for i in nonzero_idx_binary]
    number_of_nonzeros = len(nonzero_idx)
    norm = number_of_nonzeros ** (- 1 / 2)
    data = np.ones(number_of_nonzeros) * norm
    if sparse:
        return scipy.sparse.coo_matrix((data, (row, col)), shape=(2 ** len(chosen), 2 ** len(notchosen))).tocsr()
    flatrow_idx = [i * 2 ** len(notchosen) + j for i, j in zip(row, col)]
    W = np.zeros(2 ** (len(chosen) + len(notchosen)), dtype=complex)
    W[flatrow_idx] = norm
    return W.reshape((2 ** len(chosen), 2 ** len(notchosen)))


# matrix from states can be reunited if I decide to store the state directly as np.ndarray
def matrix_from_dense_state_IQFT(state: np.ndarray, chosen: list, notchosen: list):
    """
        Construct and return matrix W s.t. W.dot(W.T)==reduced density matrix for state after IQFT

    :param state: state of the system
    :param chosen: observable qubits
    :param notchosen: qubits to trace away
    :return:  W
    """

    nonzero_idx = np.flatnonzero(state)
    qubits = int(log2(len(state)))

    nonzero_idx_binary = [aux.decimal_to_binary(idx, qubits) for idx in nonzero_idx]
    row = [aux.to_decimal(aux.select_components(i, chosen)) for i in nonzero_idx_binary]
    col = [aux.to_decimal((aux.select_components(i, notchosen))) for i in nonzero_idx_binary]

    flatrow_idx = [i * 2 ** len(notchosen) + j for i, j in zip(row, col)]

    W = np.zeros(len(state), dtype=complex)
    for new_idx, old_idx in zip(flatrow_idx, nonzero_idx):
        W[new_idx] = state[old_idx]
    return W.reshape((2 ** len(chosen), 2 ** len(notchosen)))


def entanglement_entropy_from_state(state, chosen: list, sparse: bool = True) -> float:
    """
        Compute entanglement entropy of state according to chosen bipartition of qubits

    :param state:   array representing state of the system of qubits, can be scipy.sparse or numpy depending on sparse
    :param chosen:  selected qubits
    :param sparse: True if dense representation (state is np.ndarray), False if state is a scipy.sparse.coo_matrix
    :return: S
    """
    notchosen = bip.notchosen(chosen, int(log2(state.shape[0])))

    if sparse:
        W = matrix_from_sparse_modular_state(state, chosen, notchosen, sparse)
        svds = bip.sparsesvd(W, \
                             k=min(np.shape(W)) - 1, which='LM', return_singular_vectors=False)
        svds = svds ** 2
        return - np.sum([i * np.log2(i) for i in svds if i > 1e-16])

    else:
        W = matrix_from_dense_state_IQFT(state, chosen, notchosen)
        eigvals = np.array(jeigh(W.dot(W.T)))
        eigvals[eigvals < 1e-15] = 1

        return - np.sum(eigvals * np.log2(eigvals))


def entanglement_entropy_montecarlo(Y: int, N: int, maxiter: int, step: int = 100) -> list:
    """
        This function will return an approximation of bipartite entanglement entropy in Shor's Algorithm for balanced
        bipartitions. The results will be given for all the computational steps k = [1, 2L + 1]. Montecarlo methods are
        used when required. For k = [1, 2L] the computational steps consists in modular exponentiation. k = 2L + 1
        consists in the application of the IQFT on the control register.

    :param Y:       coprime of N to find the order of
    :param N:       Number to be factorized
    :param maxiter: Maximum number of iterations at which Montecarlo method stops
    :param step:    step of Montecarlo method: at least 2 * steps iteration will be computed

    :return: S:     Entanglement entropy: S[k][1] will give entropy for (k+1)-th computation steps computed
                    on different bipartitions
    """

    L = aux.lfy(N)
    # print("number of qubits: {0}+{1}".format(str(L), str(2 * L)))

    nonzeros_decimal_positions = aux.nonzeros_decimal(2 * L, N, Y)
    results = []
    current_state = 0

    ''' Modular exponentiation  '''
    for k in range(1, 2 * L + 1):
        current_state = construct_modular_state(k, L, nonzeros_decimal_positions[:2 ** k]).toarray().reshape(
            2 ** (k + L))
        considered_qubits = range(k + L)
        bipartition_size = (k + L) // 2
        combinations_considered = [bip.random_bipartition(considered_qubits, bipartition_size) for j in range(maxiter)]

        if bip.number_of_bipartitions(k + L) <= step:
            results.append([True, [entanglement_entropy_from_state(current_state, chosen, False) \
                                   for chosen in combinations(considered_qubits, bipartition_size)]])
        else:
            results.append(montecarlo_simulation(current_state, step, maxiter, combinations_considered))
            #print(str(k) + "-th computational step done \n")

    ''' IQFT '''
    current_state = applyIQFT_circuit(L, current_state)
    if bip.number_of_bipartitions(3 * L) <= step:
        results.append((True, [entanglement_entropy_from_state(current_state, chosen, False) \
                               for chosen in combinations(considered_qubits, bipartition_size)]))
    else:
        results.append(montecarlo_simulation(current_state, step, maxiter, combinations_considered))

    return results


def montecarlo_simulation(state: np.array, step: int, maxiter: int, combinations_considered: list):
    """
        Description
    :param state:                       state of the system
    :param step:                        step of Montecarlo method
    :param maxiter:                     maximum number of iteration for Montecarlo method
    :param combinations_considered:     combinations considered by the Montecarlo method
    :return:                            results as list of entropies
    """

    results = []
    for i in range(maxiter):
        current_bipartition = combinations_considered[i]
        results.append(entanglement_entropy_from_state(state, current_bipartition, False))

        # first step
        if i + 1 == step:
            previous_mean = np.mean(results)
            previous_var = np.var(results, ddof=1)
            continue

        if i + 1 % step == 0:
            current_mean = np.mean(results)
            current_var = np.var(results, ddof=1)

            tol = (i + 1) ** (- 1 / 2)
            if np.abs(previous_mean - current_mean) < tol and np.abs(previous_var - current_var) < tol:
                return True, results
            previous_mean = current_mean
            previous_var = current_var

    return False, results
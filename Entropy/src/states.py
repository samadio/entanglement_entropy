from itertools import combinations
from math import log2 as log2
from src.IQFT import *

import numpy as np
import scipy
from scipy.sparse import identity as sparse_identity
from src.auxiliary import auxiliary as aux, bipartitions as bip
from numpy.linalg import svd as numpysvd


def construct_modular_state(k: int, L: int, nonzero_elements_decimal_idx: list) -> bip.coo_matrix:
    data = np.ones(2 ** k) * 2 ** (- k / 2)
    col = np.zeros(2 ** k)
    return bip.coo_matrix((data, (nonzero_elements_decimal_idx, col)), shape=(2 ** (k + L), 1)).tocsr()

def matrix_from_state_modular(state: scipy.sparse.coo_matrix, chosen: list, notchosen: list, sparse: bool = True):
    """
        Construct and return matrix W s.t. W.dot(W.T)==reduced density matrix

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
    if sparse:
        return scipy.sparse.coo_matrix((data, (row, col)), shape=(2 ** len(chosen), 2 ** len(notchosen))).tocsr()
    flatrow_idx = [i * 2 ** len(notchosen) + j for i, j in zip(row, col)]
    W = np.zeros(2 ** (len(chosen) + len(notchosen)))
    W[flatrow_idx] = norm
    return W.reshape((2 ** len(chosen), 2 ** len(notchosen)))


def entanglement_entropy_from_state(state: scipy.sparse.coo_matrix, chosen: list, sparse: bool=True) -> float:
    """
        Compute entanglement entropy of state according to chosen bipartition of qubits

    :param state:   array representing state of the system of qubits
    :param chosen:  selected qubits
    :return: S
    """

    notchosen = bip.notchosen(chosen, int(log2(state.shape[0])))
    W = matrix_from_state_modular(state, chosen, notchosen, sparse)
    if sparse:
        svds = bip.sparsesvd(W, \
                             k=min(np.shape(W)) - 1, which='LM', return_singular_vectors=False)
    else:
        svds = numpysvd(W, compute_uv=False)

    return - np.sum([i ** 2 * 2 * np.log2(i) for i in svds if i > 1e-16])


def slicing_index(i: int, L: int) -> list:
    """auxiliary function"""
    return [i % (2 ** L) + m * 2 ** L for m in range(2 ** (2 * L))]

def entanglement_entropy(Y: int, N: int, step: int = 100) -> list:
    """
        This function will return an approximation of bipartite entanglement entropy in Shor's Algorithm for balanced
        bipartitions. The results will be given for all the computational steps k = [1, 2L + 1]. Montecarlo methods are
        used when required. For k = [1, 2L] the computational steps consists in modular exponentiation. k = 2L + 1
        consists in the application of the IQFT on the control register.

    :param N:       Number to be factorized
    :param Y:       coprime of N to find the order of
    :param step:    step of Montecarlo method: at least 2 * steps iteration will be computed
    :return: S:     Entanglement entropy: S[k][1] will give entropy for (k+1)-th computation steps computed
                    on different bipartitions
    """

    L = aux.lfy(N)
    # print("number of qubits: {0}+{1}".format(str(L), str(2 * L)))

    # TBI using period and control it's right
    nonzeros_decimal = aux.nonzeros_decimal(2 * L, N, Y)
    # print("nonzeros done")
    results = []
    current_state = 0

    ''' Modular exponentiation  '''
    for k in range(1, 2 * L + 1):
        current_state = construct_modular_state(k, L, nonzeros_decimal[:2 ** k])
        considered_qubits = range(k + L)
        bipartition_size = (k + L) // 2
        ### TO BE DELETED:
        combinations_considered = [bip.random_bipartition(range(k + L), (k + L) // 2) for j in range(200)]

        # if bip.number_of_bipartitions(k + L) <= step:
        results.append([entanglement_entropy_from_state(current_state, chosen) \
                        for chosen in combinations_considered])
        # else:
        # results.append((k, bip.montecarlo_single_k(k, Y, L, nonzero_binary, step)))
        # print(str(k) + "-th computational step done")

    ''' IQFT '''
    # FINAL STATE CAN BE COMPUTED WITHOUT THIS TENSOR PRODUCT, but probably would be less efficient: TO BE TESTED
    # tensor product: 3s but 78% of memory for N=21. Not feasible for N>=32
    # explicit calculation with for loop: 150 s, 1% memory usage
    # constructing diagonal sparse matrix: do not make sense: I should store 2**(5*L) elements. unfeasible like tensor
    # midway: even worst: 90% memory usage and a lot of time

    # t0 = time.time()
    # if L <= 5:
    #    final_state = sparse_tensordot(operator_IQFT(2 * L), sparse_identity(2 ** L)).dot(current_state)
    # else:
    final_state = applyIQFT_circuit(L, current_state)
    combinations_considered = [i for i in combinations([i for i in range(3 * L)], 3 * L // 2)][:200]
    results.append([qt.quantum_info.entropy(qt.quantum_info.partial_trace(final_state, chosen)) for chosen in
                    combinations_considered])
    return results

# midway: 13 s L=5 ,330 s for L=6
# qiskit final state: 0.25 L=4  ,1s for L=5 ,6 sec for L = 6 ,37 sec for L = 7,(311 sec  L = 8 memory 20%), after 90 min  L=9 mem swapping then error)

# _ = entanglement_entropy(13, 21, 100)

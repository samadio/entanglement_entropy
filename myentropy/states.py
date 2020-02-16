import time
from itertools import combinations
from math import log2 as log2

import numpy as np
import scipy
from scipy.sparse import identity as sparse_identity
from scipy.sparse import kron as sparse_tensordot
from myentropy import auxiliary as aux
from myentropy import bipartitions as bip


def operator_IQFT(n_qubits: int):
    """

    :param n_qubits: Integer
                     number of qubits the IQFT will be applied to

    :return: numpy.ndarray()
                    matrix form of IQFT for n qubits
    """

    linear_size = 2 ** n_qubits
    omega = - 2 * np.pi * 1j / linear_size
    return (linear_size ** (- 1 / 2)) * np.array(
        np.exp([[omega * i * k for i in range(linear_size)] for k in range(linear_size)]))


def matrix_from_state(state: scipy.sparse.coo_matrix, chosen: list, notchosen: list):
    nonzero_idx, _ = state.nonzero()
    nonzero_idx_binary = [aux.decimal_to_binary(idx, int(log2(state.shape[0]))) for idx in nonzero_idx]
    row = [aux.to_decimal(aux.select_components(i, chosen)) for i in nonzero_idx_binary]
    col = [aux.to_decimal((aux.select_components(i, notchosen))) for i in nonzero_idx_binary]
    k = int(log2(len(nonzero_idx)))
    data = np.ones(2 ** k) * (2 ** (- k / 2))
    return scipy.sparse.coo_matrix((data, (row, col)), shape=(2 ** len(chosen), 2 ** len(notchosen))).tocsc()


def entanglement_entropy_from_state(state, chosen):
    """

    :param state:       array[int]
                        binary array representing state of the system of qubits

    :param chosen:      list[int]
                        selected qubits

    :return: S:         double
                        entanglement entropy
    """
    notchosen = bip.notchosen(chosen, int(log2(state.shape[0])))
    W = matrix_from_state(state, chosen, notchosen)

    return "poivediamo"


def entanglement_entropy(Y, N, step):
    """
    This function will return an approximation of bipartite entanglement entropy in Shor's Algorithm for balanced
    bipartitions. The results will be given for all the computational steps k = [1, 2L + 1]. Montecarlo methods are
    used when required. For k = [1, 2L] the computational steps consists in modular exponentiation. k = 2L + 1
    consists in the application of the IQFT on the control register.

    :param N: Integer
              Number to be factorized

    :param Y: Integer
              coprime of N to find the order of

    :param step: Integer
              steps of Montecarlo method: at least 2 * steps iteration will be computed

    :return: S:  List of tuples
    `           Entanglement entropy: S[k][1] will give entropy for (k+1)-th computation steps computed on different bipartitions

    """

    L = aux.lfy(N)
    print("number of qubits: {0}+{1}".format(str(L), str(2 * L)))

    # TBI using period and control it's right
    nonzeros_decimal = [m * 2 ** L + (Y ** m % N) for m in range(2 ** (2 * L))]
    print("nonzeros done")
    results = []
    current_state = 0

    ''' Modular exponentiation  '''
    for k in range(1, 2 * L + 1):
        considered_qubits = range(k + L)
        bipartition_size = (k + L) // 2
        data = np.ones(2 ** k) * 2 ** (- k)
        row = nonzeros_decimal[: 2 ** k]
        col = np.zeros(2 ** k)
        current_state = bip.coo_matrix((data, (row, col)), shape=(2 ** (k + L), 1)).tocsc()

        if bip.number_of_bipartitions(k + L) <= step:
            results.append([entanglement_entropy_from_state(current_state, chosen) \
                            for chosen in combinations(considered_qubits, bipartition_size)])
        else:
            print("ciao")
            # results.append((k, bip.montecarlo_single_k(k, Y, L, nonzero_binary, step)))
        print(str(k) + "-th computational step done")

    ''' IQFT '''

    # FINAL STATE CAN BE COMPUTED WITHOUT THIS TENSOR PRODUCT, but probably would be less efficient: TO BE TESTED
    # tensor product: 3s but 78% of memory for N=21. Not feasible for N>=32
    # explicit calculation with for loop: 150 s, 1% memory usage
    operator = operator_IQFT(2 * L)
    current_state = current_state.tolil()
    t0 = time.time()
    final_state = [np.sum([operator[i // 2 ** L][j] * current_state[j * 2 ** L, 0] for j in range(2 ** (2 * L))]) for i
                   in range(2 ** (3 * L))]
    # final_state = sparse_tensordot(operator_IQFT(2 * L), sparse_identity(2 ** L)).dot(current_state)
    print("time: " + str(time.time() - t0))
    return results


print(entanglement_entropy(13, 21, 100))

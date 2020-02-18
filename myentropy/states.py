import math

import time
from itertools import combinations
from math import log2 as log2

import qiskit as qt
from qiskit.aqua.components.iqfts import Standard as IQFT
from qiskit.quantum_info import Statevector as qstate

from cmath import exp as exp
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
    return linear_size ** (- 1 / 2) * np.array( \
        np.exp([[omega * i * k for i in range(linear_size)] for k in range(linear_size)]), dtype=complex)


def operator_IQFT_row(n_qubits: int, row_idx: int) -> np.ndarray:
    """

    :param n_qubits: int
                     number of qubits the IQFT will be applied to
    :param row_idx:  int
                     row of the IQFT operator to construct

    :return: numpy.array()
             row_idx-th rows of the IQFT matrix for n qubits
    """

    linear_size = 2 ** n_qubits
    omega = - 2 * np.pi * 1j * linear_size ** (- 1)
    if row_idx >= linear_size:
        raise ValueError("The row {0} does not exist for {1} qubits".format(str(row_idx), str(n_qubits)))
    return linear_size ** (- 1 / 2) * np.array( \
        np.exp([omega * i * row_idx for i in range(linear_size)]), dtype=complex)


def matrix_from_state(state: scipy.sparse.coo_matrix, chosen: list, notchosen: list):
    nonzero_idx, _ = state.nonzero()
    nonzero_idx_binary = [aux.decimal_to_binary(idx, int(log2(state.shape[0]))) for idx in nonzero_idx]
    row = [aux.to_decimal(aux.select_components(i, chosen)) for i in nonzero_idx_binary]
    col = [aux.to_decimal((aux.select_components(i, notchosen))) for i in nonzero_idx_binary]
    k = int(log2(len(nonzero_idx)))
    data = np.ones(2 ** k) * (2 ** (- k / 2))
    return scipy.sparse.coo_matrix((data, (row, col)), shape=(2 ** len(chosen), 2 ** len(notchosen))).tocsr()


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


def slicing_index(i, L):
    return [i % (2 ** L) + m * 2 ** L for m in range(2 ** (2 * L))]


def applyIQFT(L, current_state):
    prev_control_register = qt.QuantumRegister(2 * L, 'control')
    circuit = qt.QuantumCircuit(prev_control_register, qt.QuantumRegister(L, 'target'))
    circuit.initialize(current_state.toarray().reshape(2 ** (3 * L)),[i for i in range(3 * L)])
    constructor = IQFT(2 * L)

    new_control_register = qt.QuantumRegister(2 * L, 'control')
    IQFT_circuit = qt.QuantumCircuit(new_control_register, qt.QuantumRegister(L, 'target'))
    constructor.construct_circuit(mode='circuit', circuit=IQFT_circuit, qubits=new_control_register)

    circuit = circuit.combine(IQFT_circuit)

    backend = qt.Aer.get_backend('statevector_simulator')
    final_state = qt.execute(circuit, backend).result().get_statevector()


    # return final_state
    # final_state = []
    # indexes = [slicing_index(i, L) for i in range(2 ** L)]
    # for i in range(2 ** (3 * L)):
    #   if i % (2 ** L) == 0:
    #       IQFT_row = operator_IQFT_row(2 * L, i / 2 ** L)
    #   final_state.append(np.dot(IQFT_row, current_state[indexes[i % (2 ** L)], :].toarray()))
    #    final_state.append(np.dot(IQFT_row,current_state[indexes[i% (2 ** L)]]))

    # operator = operator_IQFT(2 * L)
    # final_state = [np.sum([operator[i // 2 ** L][j] * current_state[j * 2 ** L, 0] for j in range(2 ** (2 * L))]) \
    #               for i in range(2 ** (3 * L))]
    # final_state = [np.sum([operator[i // 2 ** L][j] * current_state[j * 2 ** L, 0] \
    #                           if j * 2 ** L in current_state.nonzero()[0] else 0 for j in range(2 ** (2 * L))]) \
    #               for i in range(2 ** (3 * L))]

    # naive implementation: no memory usage but O (2 ** (5 * L)) computations
    # omega = exp(2 * np.pi * 1j * 2 ** (- 2 * L))
    # normalization_constant = 2 ** (- L)
    # final_state = normalization_constant * np.array([np.sum([(omega ** (i * k)) * current_state[k * 2 ** L, 0] \
    #                                                             if k * 2 ** L in current_state.nonzero()[0] else 0 for \
    #                                                         k in range(2 ** (2 * L))]) \
    #                                                 for i in range(2 ** (3 * L))])

    # probably to be deleted
    # target_state_sequence = np.array([list(aux.decimal_to_binary((Y ** m) % N, L)) for m in range(2 ** (2 * L))],dtype=np.short)
    # target_state_sequence = bip.coo_matrix(target_state_sequence).tocsr()
    # print(target_state_sequence.toarray())

    # final_state = np.zeros(2 ** (3 * L))
    # for l in range (2 ** (2 * L)):

    return 0


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
    nonzeros_decimal = [m * 2 ** L + ((Y ** m) % N) for m in range(2 ** (2 * L))]
    print("nonzeros done")
    results = []
    current_state = 0

    ''' Modular exponentiation  '''
    for k in range(1, 2 * L + 1):
        considered_qubits = range(k + L)
        bipartition_size = (k + L) // 2
        data = np.ones(2 ** k) * 2 ** (- k / 2)
        row = nonzeros_decimal[: 2 ** k]
        col = np.zeros(2 ** k)
        current_state = bip.coo_matrix((data, (row, col)), shape=(2 ** (k + L), 1)).tocsr()

        if bip.number_of_bipartitions(k + L) <= step:
            results.append([entanglement_entropy_from_state(current_state, chosen) \
                            for chosen in combinations(considered_qubits, bipartition_size)])
        else:
            # results.append((k, bip.montecarlo_single_k(k, Y, L, nonzero_binary, step)))
            print(str(k) + "-th computational step done")

    ''' IQFT '''
    # FINAL STATE CAN BE COMPUTED WITHOUT THIS TENSOR PRODUCT, but probably would be less efficient: TO BE TESTED
    # tensor product: 3s but 78% of memory for N=21. Not feasible for N>=32
    # explicit calculation with for loop: 150 s, 1% memory usage
    # constructing diagonal sparse matrix: do not make sense: I should store 2**(5*L) elements. unfeasible like tensor
    # midway: even worst: 90% memory usage and a lot of time

    t0 = time.time()
    # if L <= 5:
    #    final_state = sparse_tensordot(operator_IQFT(2 * L), sparse_identity(2 ** L)).dot(current_state)
    # else:
    final_state = applyIQFT(L, current_state)
    print("time: " + str(time.time() - t0))

    return results


# midway: 13 s L=5 ,330 s for L=6
# qiskit: 0.25 ,1s for L=5 ,6 sec for L = 6 ,37 sec for L = 7,(311 sec  L = 8 memory 20%),
_ = entanglement_entropy(13, 15, 100)

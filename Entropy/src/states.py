from itertools import combinations
from math import log2 as log2

import qiskit as qt
from qiskit.aqua.components.iqfts import Standard as IQFT

import numpy as np
import scipy
from scipy.sparse import identity as sparse_identity
from auxiliary import auxiliary as aux, bipartitions as bip


def construct_state(k, L, row):
    data = np.ones(2 ** k) * 2 ** (- k / 2)
    col = np.zeros(2 ** k)
    return bip.coo_matrix((data, (row, col)), shape=(2 ** (k + L), 1)).tocsr()


def operator_IQFT(n: int) -> np.array:
    """
        Build and return the IQFT matrix for n qubits

    :param n:   number of qubits the IQFT will be applied to
    :return:    matrix form of IQFT for n qubits
    """

    linear_size = 2 ** n
    omega = - 2 * np.pi * 1j / linear_size
    return linear_size ** (- 1 / 2) * np.array( \
        np.exp([[omega * i * k for i in range(linear_size)] for k in range(linear_size)]), dtype=complex)


def operator_IQFT_row(n: int, i: int) -> np.array:
    """
        Build and return i-th row of IQFT matrix acting on n qubits

    :param n:    number of qubits the IQFT will be applied to
    :param i:    row of the IQFT operator to construct
    :return:  i-th row of the IQFT matrix for n qubits
    """

    linear_size = 2 ** n
    omega = - 2 * np.pi * 1j * linear_size ** (- 1)
    if i >= linear_size:
        raise ValueError("The row {0} does not exist for {1} qubits".format(str(i), str(n)))
    return linear_size ** (- 1 / 2) * np.array( \
        np.exp([omega * i * i for i in range(linear_size)]), dtype=complex)


def matrix_from_state(state: scipy.sparse.coo_matrix, chosen: list, notchosen: list) -> scipy.sparse.coo_matrix:
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
    data = np.ones(number_of_nonzeros) * number_of_nonzeros ** (- 1 / 2)
    return scipy.sparse.coo_matrix((data, (row, col)), shape=(2 ** len(chosen), 2 ** len(notchosen))).tocsr()


def entanglement_entropy_from_state(state: scipy.sparse.coo_matrix, chosen: list) -> float:
    """
        Compute entanglement entropy of state according to chosen bipartition of qubits

    :param state:   array representing state of the system of qubits
    :param chosen:  selected qubits
    :return: S
    """

    notchosen = bip.notchosen(chosen, int(log2(state.shape[0])))
    W = matrix_from_state(state, chosen, notchosen)
    eigs = bip.sparsesvd(W, \
                         k=min(np.shape(W)) - 1, which='LM', return_singular_vectors=False)

    return - np.sum([i ** 2 * 2 * np.log2(i) for i in eigs if i > 1e-16])


def slicing_index(i: int, L: int) -> list:
    """auxiliary function"""
    return [i % (2 ** L) + m * 2 ** L for m in range(2 ** (2 * L))]


def applyIQFT_circuit(L: int, current_state: scipy.sparse.coo_matrix) -> scipy.sparse.coo_matrix:
    """
        Apply IQFT on target register of current state and returns final state

    :param L: number of qubits in target register
    :param current_state: state to apply IQFT on
    :return: state after IQFT
            """
    control_register = qt.QuantumRegister(2 * L, 'control')
    target_register = qt.QuantumRegister(L, 'target')
    circuit = qt.QuantumCircuit(control_register, target_register)
    circuit.initialize(current_state.toarray().reshape(2 ** (3 * L)), range(3 * L))

    constructor = IQFT(control_register)
    IQFT_circuit = qt.QuantumCircuit(control_register, target_register)
    constructor.construct_circuit(mode='circuit', circuit=IQFT_circuit, qubits=control_register)

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

    return final_state


def applyIQFT(L: int, current_state: qt.quantum_info.Statevector) -> qt.quantum_info.Statevector:
    """
        Apply IQFT on target register of current state and returns final state

    :param L: number of qubits in target register
    :param current_state: state to apply IQFT on
    :return: state after IQFT
            """

    constructor = IQFT(2 * L)

    control_register = qt.QuantumRegister(2 * L, 'control')
    IQFT_operator = qt.quantum_info.Operator( \
        constructor.construct_circuit(mode='matrix', qubits=control_register) \
        )

    final_state = current_state.evolve(IQFT_operator, range(2 * L))
    return final_state


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
        current_state = construct_state(k, L, nonzeros_decimal[:2 ** k])
        considered_qubits = range(k + L)
        bipartition_size = (k + L) // 2
        ### TO BE DELETED:
        combinations_considered = [bip.random_bipartition(range(k + L), (k+L) // 2) for j in range(200)]


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
    results.append([qt.quantum_info.entropy(qt.quantum_info.partial_trace(final_state, chosen)) for chosen in combinations_considered])
    return results

# midway: 13 s L=5 ,330 s for L=6
# qiskit final state: 0.25 L=4  ,1s for L=5 ,6 sec for L = 6 ,37 sec for L = 7,(311 sec  L = 8 memory 20%), after 90 min  L=9 mem swapping then error)

# _ = entanglement_entropy(13, 21, 100)

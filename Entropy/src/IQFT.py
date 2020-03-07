from scipy.sparse import coo_matrix, identity
from scipy.sparse import kron as tensor
import numpy as np

import qiskit as qt
from qiskit.aqua.circuits import FourierTransformCircuits as IQFT


def applyIQFT_circuit(L: int, current_state: coo_matrix) -> np.ndarray:
    """
        Apply IQFT on target register of current state and returns final state

    :param L: number of qubits in target register
    :param current_state: state to apply IQFT on
    :return: state after IQFT
            """
    circuit = qt.QuantumCircuit(3 * L)
    circuit.initialize(current_state.toarray().reshape(2 ** (3 * L)), [i for i in reversed(circuit.qubits)])

    IQFT.construct_circuit(circuit=circuit, qubits=circuit.qubits[:2 * L], inverse=True)

    backend = qt.Aer.get_backend('statevector_simulator')
    final_state = qt.execute(circuit, backend).result().get_statevector()

    return final_state


def apply_IQFT(L: int, current_state: coo_matrix) -> np.ndarray:
    control_qubits = 2 * L
    target_qubits = L
    ide = identity(2 ** target_qubits)
    '''final_state = []

    for i in range(2 ** (2 * L)):
        IQFT_row = operator_IQFT_row(2 * L, i)
        IQFT_operator = tensor(IQFT_row, ide)
        print(IQFT_row.toarray())

        temp_res = IQFT_operator.dot(current_state).toarray().flatten()
        final_state.append(temp_res)
        '''

    linear_size = 2 ** control_qubits
    omega = - 2 * np.pi * 1j / linear_size
    TESTIQFT = linear_size ** (- 1 / 2) * np.exp(
        [[omega * i * k for k in range(linear_size)] for i in range(linear_size)], dtype=complex)

    TESTIQFT = tensor(TESTIQFT, ide)
    return TESTIQFT.dot(current_state).toarray().flatten()

    return np.array(final_state).flatten()


def operator_IQFT_row(n: int, i: int) -> np.ndarray:
    """
        Build and return i-th row of IQFT matrix acting on n qubits
    :param n:    number of qubits the IQFT will be applied to
    :param i:    row of the IQFT operator to construct
    :return:  i-th row of the IQFT matrix for n qubits
    """

    linear_size = 2 ** n
    omega = - 2 * np.pi * 1j / linear_size

    if i >= linear_size:
        raise ValueError("The row {0} does not exist for {1} qubits".format(str(i), str(n)))

    return linear_size ** (- 1 / 2) * np.array(
        np.exp([omega * i * k for k in range(linear_size)]), dtype=complex)

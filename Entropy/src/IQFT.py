from scipy.sparse import coo_matrix, identity
from scipy.sparse import kron as tensor
import numpy as np

import qiskit as qt
from qiskit.aqua.circuits import FourierTransformCircuits as QFT


def applyIQFT_circuit(L: int, current_state: coo_matrix) -> np.ndarray:
    """
        Apply IQFT on control register of current state and returns final state

    :param L: number of qubits in target register
    :param current_state: state to apply IQFT on
    :return: state after IQFT
    """
    circuit = qt.QuantumCircuit(3 * L)
    circuit.initialize(current_state.toarray().reshape(2 ** (3 * L)), [i for i in reversed(circuit.qubits)])

    QFT.construct_circuit(circuit=circuit, qubits=circuit.qubits[:2 * L], inverse=True, do_swaps=True)

    backend = qt.Aer.get_backend('statevector_simulator')
    final_state = qt.execute(circuit, backend, shots=1).result().get_statevector()

    return final_state


def apply_IQFT_huge(L: int, current_state: coo_matrix) -> np.ndarray:
    control_qubits = 2 * L
    target_qubits = L

    return tensor(operator_IQFT(control_qubits), identity(target_qubits)).dot(current_state)

    '''ide = identity(2 ** target_qubits)

    result = np.fromfunction(lambda i: IQFT_auxiliary(i, control_qubits, ide, current_state), (2 ** control_qubits,),
                             dtype=complex)
    return result.flatten()
    more readable version
    for i in range(2 ** (2 * L)):
        IQFT_row = operator_IQFT_row(control_qubits, i)
        IQFT_operator_global = tensor(IQFT_row, ide)

        temp_res = IQFT_operator_global.dot(current_state).toarray().flatten()
        final_state.append(temp_res)

    return np.array(final_state).flatten()
    '''


def operator_IQFT_row(n: int, i: int, inverse: bool = True) -> np.ndarray:
    """
        Build and return i-th row of IQFT matrix acting on n qubits
    :param inverse: if True IQFT is returned, otherwise QFT
    :param n:    number of qubits the IQFT will be applied to
    :param i:    row of the IQFT operator to construct
    :return:  i-th row of the IQFT matrix for n qubits
    """

    linear_size = 2 ** n
    omega = 2 * np.pi * 1j / linear_size
    if inverse:
        omega = -omega

    if i >= linear_size:
        raise ValueError("The row {0} does not exist for {1} qubits".format(str(i), str(n)))

    return linear_size ** (- 1 / 2) * np.array(
        np.exp([omega * i * k for k in range(linear_size)]), dtype=complex)


def operator_IQFT(n: int, inverse: bool = True) -> np.ndarray:
    """
        Build and return i-th row of IQFT matrix acting on n qubits
    :param inverse: if True IQFT operator is returned, otherwise QFT
    :param n:    number of qubits the IQFT will be applied to
    :return:  i-th row of the IQFT matrix for n qubits
    """

    return np.array([operator_IQFT_row(n, i, inverse) for i in range(2 ** n)])
import numpy as np
import qiskit as qt
from scipy.sparse import coo_matrix, identity
from scipy.sparse import kron as tensor

from auxiliary import auxiliary as aux
from qiskit.execute import execute
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.aqua.circuits import FourierTransformCircuits as QFT
#from qiskit_qcgpu_provider import QCGPUProvider

def apply_IQFT(L: int, current_state: np.array) -> np.ndarray:
    """
        apply IQFT to control target using different methods depending on number of qubits
    :param L: number of qubits in target register
    :param current_state: current_state of the system
    :return: state after IQFT on control register
    """
    if L < 6:
        return applyIQFT_huge(L, current_state).reshape(2 ** (3 * L))
    else:
        return applyIQFT_circuit(L, current_state)


def applyIQFT_circuit(L: int, current_state: np.ndarray) -> np.ndarray:
    """
        Apply IQFT on control register of current state and returns final state using qiskit circuit

    :param L: number of qubits in target register
    :param current_state: state to apply IQFT on
    :return: state after IQFT
    """

    circuit = QuantumCircuit(3 * L)
    circuit.initialize(current_state.reshape(2 ** (3 * L)), circuit.qubits)
    circuit = QFT.construct_circuit(circuit=circuit, qubits=circuit.qubits[L:3 * L], inverse=True, do_swaps=True)

    backend = qt.Aer.get_backend('statevector_simulator')
    # backend = QCGPUProvider().get_backend('statevector_simulator')
    final_state = execute(circuit, backend, shots=1).result().get_statevector()

    return final_state

def applyIQFT_huge(L: int, current_state: coo_matrix) -> np.ndarray:
    """
           Apply IQFT on control register of current state and returns final state using tensor operator

       :param L: number of qubits in target register
       :param current_state: state to apply IQFT on
       :return: state after IQFT
    """
    control_qubits = 2 * L
    target_qubits = L

    return tensor(operator_IQFT(control_qubits), identity(2 ** target_qubits)).dot(current_state)


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

# --------------------------------------------------------------------------------------------------------
#from here on, they've not been used

def operator_IQFT(n: int, inverse: bool = True) -> np.ndarray:
    """
        Build and return i-th row of IQFT matrix acting on n qubits
    :param inverse: if True IQFT operator is returned, otherwise QFT
    :param n:    number of qubits the IQFT will be applied to
    :return:  i-th row of the IQFT matrix for n qubits
    """

    return np.array([operator_IQFT_row(n, i, inverse) for i in range(2 ** n)])


def invert_qubits_state(state: coo_matrix, length: int) -> coo_matrix:
    new_idx = range(2 ** length)
    result = np.array(list(map(lambda i: aux.to_decimal(aux.decimal_to_binary(i, length)[::-1]), new_idx)),
                      dtype=np.int64)
    return state.toarray().reshape((2 ** length), )[result]

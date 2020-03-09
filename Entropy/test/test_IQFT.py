from unittest import TestCase
from IQFT import *
from states import *
from math import sqrt
from qutip.qip.algorithms.qft import qft
from qutip import tensor, qeye
from time import time


class Test(TestCase):

    def test_IQFT_row_correct(self):
        nqubits = 3
        linear_size = 2 ** nqubits
        IQFT_row = operator_IQFT_row(nqubits, 0)
        np.testing.assert_array_almost_equal(IQFT_row, np.ones(linear_size) / sqrt(linear_size))
        nqubits = 2

        IQFT_row = operator_IQFT_row(nqubits, 2).round(15).tolist()
        self.assertListEqual([1 / 2, -1 / 2, 1 / 2, -1 / 2], IQFT_row)

        IQFT_row = operator_IQFT_row(nqubits, 1).astype(complex).tolist()
        np.testing.assert_array_almost_equal(1 / 2 * np.array([1, -1j, -1, 1j]), IQFT_row)

    def test_qiskit_IQFT_correct(self):
        control_state = np.ones(4) / 2
        target_state = np.array([1, 0])  # zero state
        state = np.kron(control_state, target_state).reshape(8, 1)
        state = coo_matrix(state, shape=(8, 1))
        qfinal = applyIQFT_circuit(1, state).round(decimals=6)

        expected_control_state = np.array([1, 0, 0, 0])
        expected = np.kron(expected_control_state, target_state)

        np.testing.assert_array_almost_equal(qfinal, expected)

    def test_myQFT_correct(self):
        nqubits = 2
        linear_size = 2 ** nqubits
        omega = 2 * np.pi * 1j / linear_size
        benchIQFT = qft(nqubits)
        myIQFT = linear_size ** (- 1 / 2) * np.exp(
            [[omega * i * k for k in range(linear_size)] for i in range(linear_size)], dtype=complex)
        np.testing.assert_array_almost_equal(benchIQFT, myIQFT)

    def test_IQFT_global_correct(self):
        target_qubits = 1
        contr_qubits = 2 * target_qubits
        linear_size = 2 ** contr_qubits
        omega = 2 * np.pi * 1j / linear_size
        benchIQFT = tensor(qft(contr_qubits), qeye(2 ** target_qubits))
        print(benchIQFT.shape)

        myIQFT = linear_size ** (- 1 / 2) * np.exp(
            [[omega * i * k for k in range(linear_size)] for i in range(linear_size)], dtype=complex)

        myIQFT = np.kron(myIQFT, np.eye(2 ** target_qubits))
        np.testing.assert_array_almost_equal(benchIQFT, myIQFT)

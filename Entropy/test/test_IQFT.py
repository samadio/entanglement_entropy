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
        N = 21
        Y = 13
        L = 5
        k = 2 * L
        nonzeros = aux.nonzeros_decimal(k, Y, N)
        state = construct_modular_state(k, L, nonzeros)
        qfinal = applyIQFT_circuit(L, state)

        IQFT_matrix_2L = coo_matrix(operator_IQFT(2 * L))
        operator = scipy.sparse.kron(IQFT_matrix_2L, scipy.eye(2 ** L))
        print(operator.shape)
        print("Percentage of nonzeros in operator:")
        print(len(operator.nonzero()[0]) == 2 ** (5 * L))
        exact = operator.dot(state)
        print("Percentage of nonzeros in state:")
        print(100 * len(exact.nonzero()[0]) / 2 ** (3 * L))
        exact = exact.toarray().reshape(2 ** (3 * L))

        exact[np.abs(exact) < 1e-14] = 0
        qfinal[np.abs(qfinal) < 1e-14] = 0
        np.testing.assert_array_almost_equal(exact, qfinal)

    def test_myIQFT_operator_correct(self):
        nqubits = 4
        benchIQFT = qft(nqubits)
        myIQFT = operator_IQFT(nqubits, False)
        np.testing.assert_array_almost_equal(benchIQFT, myIQFT)

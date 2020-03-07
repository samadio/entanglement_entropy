from unittest import TestCase
from IQFT import *
from states import *
from math import sqrt

class Test(TestCase):

    def test_IQFT_row_correct(self):
        nqubits = 3
        linear_size = 2 ** nqubits
        IQFT_row = operator_IQFT_row(nqubits, 0)
        np.testing.assert_array_almost_equal(IQFT_row, np.ones(linear_size) / sqrt(linear_size))
        nqubits = 2

        IQFT_row = operator_IQFT_row(nqubits, 2).round(15).tolist()
        self.assertListEqual([1/2,-1/2,1/2,-1/2],IQFT_row)

        IQFT_row = operator_IQFT_row(nqubits, 1).astype(complex).tolist()
        np.testing.assert_array_almost_equal(1/2 * np.array([1,-1j,-1,1j]),IQFT_row)


    def test_my_final_state_equals_qiskit(self):
        N = 2
        Y = 1
        L = 1
        k = 2 * L
        nonzero_elements = aux.nonzeros_decimal(k, N, Y)
        state = construct_state(k, L, nonzero_elements)


        my_final = apply_IQFT(L, state).round(decimals=6)
        qfinal = applyIQFT_circuit(L, state).round(decimals=6)

        print(my_final)
        print(qfinal)

        np.testing.assert_array_almost_equal(my_final, qfinal)
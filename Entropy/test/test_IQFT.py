from unittest import TestCase
from states import *
from math import sqrt
from qutip.qip.algorithms.qft import qft
import scipy


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

        N = 25
        max_test_size = 30
        random_coprimes = np.unique([aux.coprime(N) for i in range(max_test_size)])
        for Y in random_coprimes:
            L = aux.lfy(N)
            k = 2 * L
            nonzeros = aux.nonzeros_decimal(k, Y, N)
            state = construct_modular_state(k, L, nonzeros)
            qfinal = applyIQFT_circuit(L, state.toarray())

            IQFT_matrix_2L = coo_matrix(operator_IQFT(2 * L))
            operator = scipy.sparse.kron(IQFT_matrix_2L, scipy.eye(2 ** L))
            exact = operator.dot(state)
            exact = exact.toarray().reshape(2 ** (3 * L))

            np.testing.assert_array_almost_equal(exact, qfinal)

    def test_myIQFT_operator_correct(self):
        nqubits = 4
        benchIQFT = qft(nqubits)
        myIQFT = operator_IQFT(nqubits, False)
        np.testing.assert_array_almost_equal(benchIQFT, myIQFT)

    def test_tensor_zero(self):
        matrix = scipy.sparse.random(2 ** 5, 2 ** 5)

        new_matrix = scipy.sparse.kron(scipy.eye(1), matrix).toarray()
        np.testing.assert_array_almost_equal(matrix.toarray(), new_matrix)

    def test_qiskit_order(self):
        simulator = qt.Aer.get_backend('statevector_simulator')

        L = 2

        state = np.zeros(2 ** (3 * L))
        #state = 00..10
        state[2] = 1

        circuit = qt.QuantumCircuit(3 * L)
        circuit.initialize(state, circuit.qubits)
        circuit.h(1)
        returned_state = qt.execute(circuit, simulator, shots=1).result().get_statevector()
        exact = np.zeros(2 ** (3 * L))
        exact[0] = 1/sqrt(2)
        exact[2] = - exact[0]
        np.testing.assert_array_almost_equal(returned_state, exact)

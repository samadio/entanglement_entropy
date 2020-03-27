from unittest import TestCase

from src.auxiliary.bipartitions import random_bipartition, notchosen
from src.auxiliary.bipartitions import entropy as binentropy
from src.states import *
from qiskit.quantum_info import partial_trace, entropy, Statevector


class Test(TestCase):

    def test_construct_state_normalized(self):
        k = 3
        L = 4
        N = 13
        Y = 6
        state = construct_modular_state(k, L, aux.nonzeros_decimal(k, N, Y)).toarray()
        self.assertAlmostEqual(np.sum([i ** 2 for i in state]), 1, places=13)

    def test_construct_state_correct(self):
        k = 6
        L = 3
        N = 7
        Y = 2
        state = construct_modular_state(k, L, aux.nonzeros_decimal(k, N, Y))
        self.assertListEqual(state.nonzero()[0].tolist(), aux.nonzeros_decimal(k, N, Y))

    def test_myW_are_equals(self):
        k = 12
        L = 6
        N = 33
        Y = 17
        nonzeros_decimal = aux.nonzeros_decimal(k, N, Y)
        nonzeros_binary = [aux.decimal_to_binary(i, k + L) for i in nonzeros_decimal]
        state = construct_modular_state(k, L, nonzeros_decimal)
        tries = 10
        for i in range(tries):
            chosen = random_bipartition([i for i in range(k + L)], (k + L) // 2)
            W_binary = bip.create_w_from_binary(chosen, bip.notchosen(chosen, k + L), nonzeros_binary).toarray()
            W_state = matrix_from_state_modular(state, chosen, notchosen(chosen, k + L)).toarray()
            self.assertListEqual(W_binary.tolist(), W_state.tolist())

    def tests_my_entropies_are_equal(self):
        k = 12
        L = 6
        N = 33
        Y = 23
        number_of_qubits = k + L
        nonzeros_decimal = aux.nonzeros_decimal(k, N, Y)
        nonzeros_binary = [aux.decimal_to_binary(i, number_of_qubits) for i in nonzeros_decimal]
        state = construct_modular_state(k, L, nonzeros_decimal)
        tries = 20
        for i in range(tries):
            chosen = random_bipartition(range(number_of_qubits), number_of_qubits // 2)
            state_entropy = entanglement_entropy_from_state(state, chosen)
            binary_entropy = binentropy(k, L, chosen, nonzeros_binary)
            self.assertTrue(np.abs(binary_entropy - state_entropy) < 1e-14)

    def test_myentropy_equal_qentropy(self):
        k = 8
        L = 5
        Y = 13
        N = 21
        number_of_qubits = k + L
        nonzeros_decimal = aux.nonzeros_decimal(k, N, Y)
        state = construct_modular_state(k, L, nonzeros_decimal)
        qstate = Statevector(state.toarray())
        tries = 30
        for i in range(tries):
            chosen_qubits = random_bipartition(range(number_of_qubits), number_of_qubits // 2)
            notchosen_qubits = notchosen(chosen_qubits, number_of_qubits)
            myentropy = entanglement_entropy_from_state(state, chosen_qubits)
            qentropy = entropy(partial_trace(qstate, [number_of_qubits - 1 - i for i in notchosen_qubits]))
            self.assertTrue(np.abs(myentropy - qentropy) < 1e-14)

    def test_IQFT_entropy_equals_qiskit(self):
        Y = 13
        N = 21
        L = aux.lfy(N)
        k = 2 * L
        number_of_qubits = 3 * L

        state = construct_modular_state(k, L, aux.nonzeros_decimal(k, Y, N))
        state = apply_IQFT(L, state)

        tries = 30
        for i in range(tries):
            chosen_qubits = random_bipartition(range(number_of_qubits), number_of_qubits // 2)
            notchosen_qubits = notchosen(chosen_qubits, number_of_qubits)
            myentropy = entanglement_entropy_from_state(state, chosen_qubits, False)
            qentropy = entropy(partial_trace(Statevector(state), [number_of_qubits - 1 - i for i in notchosen_qubits]))
            self.assertTrue(np.abs(myentropy - qentropy) < 1e-13)

    def test_max_memory(self):
        for L in range(5,13):
            state = np.ones(2 ** (3 * L + 1))
            print(str(L) + "it's fine")

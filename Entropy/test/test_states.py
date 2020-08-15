from unittest import TestCase
from auxiliary.bipartitions import random_bipartition, notchosen
from auxiliary.bipartitions import entropy as binentropy
from states import *
from qiskit.quantum_info import partial_trace, entropy, Statevector
from jax.numpy.linalg import svd as jsvd


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
        k = 10
        Y = 13
        N = 33
        L = aux.lfy(N)

        number_of_qubits = k + L
        nonzeros_decimal = aux.nonzeros_decimal(k, N, Y)
        state = construct_modular_state(k, L, nonzeros_decimal)
        qstate = Statevector(state.toarray())
        tries = 100
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
            self.assertTrue(np.abs(myentropy - qentropy) < 1e-5)

    def test_max_memory(self):
        for L in range(5, 13):
            state = np.ones(2 ** (3 * L + 1))
            print(str(L) + "it's fine")

    def test_projected_entropy(self):
        Y = 13
        N = 21
        L = aux.lfy(N)
        k = L
        number_of_qubits = k + L

        proj_state = construct_modular_state(k, L, aux.nonzeros_decimal(k, Y, N)).toarray().flatten()

        state = np.zeros(2 ** (3 * L))
        state[:2 ** (k + L)] = proj_state.flatten()

        tries = 100
        for i in range(tries):
            chosen_qubits = random_bipartition(range(number_of_qubits), number_of_qubits // 2)
            proj_entr = entanglement_entropy_from_state(proj_state, chosen_qubits, False)
            entr = entanglement_entropy_from_state(state, [2 * L - k + i for i in chosen_qubits], False)

            np.testing.assert_almost_equal(proj_entr, entr)

    def test_svd_jax(self):
        Y = 13
        numb = [22]
        tries = 100

        for N in numb:
            eigval_entr = []
            jax_entr = []
            L = aux.lfy(N)
            k = 2 * L
            number_of_qubits = k + L
            state = construct_modular_state(k, L, aux.nonzeros_decimal(k, Y, N))
            chosens = [random_bipartition(range(number_of_qubits), number_of_qubits // 2) for i in range(tries)]
            for i in range(tries):
                chosen = chosens[i]
                notchosen = bip.notchosen(chosen, k + L)
                W = matrix_from_state_modular(state, chosen, notchosen, False)
                jsv = np.array(jsvd(W, compute_uv=False), dtype=float)
                jsv2 = jsv ** 2
                jsv2[jsv2 < 1e-15] = 1
                jax_entr.append(np.sum(jsv2 * np.log2(jsv2)))
                l = np.linalg.eigvalsh(W.dot(W.T))
                l[l < 1e-15] = 1
                eigval_entr.append(np.sum(l * np.log2(l)))
            np.testing.assert_array_almost_equal(eigval_entr, jax_entr, decimal=5)
            #jax is precise up to 1e-6

'''
    def test_svd_eigvals(self):
        Y = 13
        numb = [21, 33, 66]
        tries = 100

        for N in numb:
            eigval_entr = []
            svd_entr = []
            L = aux.lfy(N)
            k = 2 * L
            number_of_qubits = k + L
            state = construct_modular_state(k, L, aux.nonzeros_decimal(k, Y, N))
            chosens = [random_bipartition(range(number_of_qubits), number_of_qubits // 2) for i in range(tries)]
            for i in range(tries):
                chosen = chosens[i]
                notchosen = bip.notchosen(chosen, k + L)
                W = matrix_from_state_modular(state, chosen, notchosen, False)
                sv = numpysvd(W, compute_uv=False)
                sv2 = sv ** 2
                sv2[sv2 < 1e-15] = 1
                svd_entr.append(np.sum(sv2 * np.log2(sv2)))
                l = np.linalg.eigvalsh(W.dot(W.T))
                l[l < 1e-15] = 1
                eigval_entr.append(np.sum(l * np.log2(l)))
            np.testing.assert_array_almost_equal(eigval_entr, svd_entr, decimal=12)
'''

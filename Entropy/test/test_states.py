from unittest import TestCase
from auxiliary.bipartitions import random_bipartition, notchosen
from qiskit.quantum_info import partial_trace, entropy, Statevector
from jax.numpy.linalg import svd as jsvd
from numpy.linalg import svd as numpysvd
from entropies import *


class Test(TestCase):

    def test_construct_state_normalized(self):
        k = 3
        L = 4
        N = 13
        Y = 6
        state = construct_modular_state(k, L, aux.nonzeros_decimal(k, N, Y)).toarray()
        np.testing.assert_almost_equal(np.sum(state ** 2), 1.)

    def test_construct_state_correct(self):
        k = 6
        L = 3
        N = 7
        Y = 2
        state = construct_modular_state(k, L, aux.nonzeros_decimal(k, N, Y))
        self.assertListEqual(state.nonzero()[0].tolist(), aux.nonzeros_decimal(k, N, Y))

    def test_sparse_entropy_equal_qiskit(self):
        N = 33
        Y = aux.coprime(N)
        L = aux.lfy(N)
        k = 2 * L
        number_of_qubits = k + L
        nonzeros_decimal = aux.nonzeros_decimal(k, N, Y)
        state = construct_modular_state(k, L, nonzeros_decimal)
        tries = 200
        for i in range(tries):
            chosen = random_bipartition(range(number_of_qubits), number_of_qubits // 2)
            notchosen = bip.notchosen(chosen, number_of_qubits)
            state_entropy = entanglement_entropy_from_state(state, chosen, sparse=True)
            qentropy = entropy(partial_trace(
                Statevector(state.toarray().flatten()),
                [k+L-1-i for i in notchosen]))

            self.assertTrue(np.abs(qentropy - state_entropy) < 1e-12)

    def test_IQFT_entropy_equals_qiskit(self):
        Y = 13
        N = 28
        L = aux.lfy(N)
        k = 2 * L
        number_of_qubits = k + L

        state = construct_modular_state(k, L, aux.nonzeros_decimal(k, Y, N))
        state = apply_IQFT(L, state.toarray())

        tries = 30
        for i in range(tries):
            chosen_qubits = random_bipartition(range(number_of_qubits), number_of_qubits // 2)
            notchosen_qubits = notchosen(chosen_qubits, number_of_qubits)
            myentropy = entanglement_entropy_from_state(state, chosen_qubits, sparse=False, gpu=False )
            qentropy = entropy(partial_trace(Statevector(state), notchosen_qubits))
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
            proj_entr = entanglement_entropy_from_state(proj_state, chosen_qubits, sparse=False, gpu=False)
            entr = entanglement_entropy_from_state(state, chosen_qubits, sparse=False, gpu=False)

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
            state = construct_modular_state(k, L, aux.nonzeros_decimal(k, Y, N)).toarray().flatten()
            chosens = [random_bipartition(range(number_of_qubits), number_of_qubits // 2) for i in range(tries)]
            for i in range(tries):
                chosen = chosens[i]
                notchosen = bip.notchosen(chosen, k + L)
                W = W_from_state_numba(state, chosen, notchosen)
                jsv = np.array(jsvd(W, compute_uv=False), dtype=float)
                jsv2 = jsv ** 2
                jsv2[jsv2 < 1e-15] = 1
                jax_entr.append(np.sum(jsv2 * np.log2(jsv2)))
                l = np.linalg.eigvalsh(W.dot(W.T))
                l[l < 1e-15] = 1
                eigval_entr.append(np.sum(l * np.log2(l)))
            np.testing.assert_array_almost_equal(eigval_entr, jax_entr, decimal=5)
            #jax is precise up to 1e-6


    def test_svd_eigvals(self):
        Y = 13
        numb = [21, 33, 66]
        tries = 50

        for N in numb:
            eigval_entr = []
            svd_entr = []
            L = aux.lfy(N)
            k = 2 * L
            number_of_qubits = k + L
            state = construct_modular_state(k, L, aux.nonzeros_decimal(k, Y, N)).toarray().flatten()
            chosens = [random_bipartition(range(number_of_qubits), number_of_qubits // 2) for i in range(tries)]
            for i in range(tries):
                chosen = chosens[i]
                notchosen = bip.notchosen(chosen, k + L)
                W = W_from_state_numba(state, chosen, notchosen)
                sv = numpysvd(W, compute_uv=False)
                sv2 = sv ** 2
                sv2[sv2 < 1e-15] = 1
                svd_entr.append(np.sum(sv2 * np.log2(sv2)))
                l = np.linalg.eigvalsh(W.dot(W.T))
                l[l < 1e-15] = 1
                eigval_entr.append(np.sum(l * np.log2(l)))
            np.testing.assert_array_almost_equal(eigval_entr, svd_entr, decimal=12)

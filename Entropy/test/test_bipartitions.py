from unittest import TestCase

from auxiliary.bipartitions import *
import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace
from qiskit.quantum_info import entropy as qentropy
from numpy.linalg import svd as numpysvd


class Test(TestCase):
    def test_notchosen(self):
        selected = [i * 2 for i in range(50)]
        self.assertListEqual(notchosen(selected, 100), [i * 2 + 1 for i in range(50)])
        self.assertListEqual(notchosen([1, 3], 4), [0, 2])
        with self.assertRaises(ValueError):
            notchosen([1, 3, 5, 7], 4)

    def test_create_w_from_binary(self):
        # psi=[0,1/2,1/2,0,0,1/2,0,1/2], chosen=[0,2] -> nonzeros=[1,2,5,7], nonzero_binary= ['001','010','101','111']
        # rows = ['01'-> 1, 0, 3, 3]          cols= [0,1,0,1]
        # W = 1 / 2 [ [0,1] , [1,0], [0,0], [1,1] ]
        state = np.array([0, 1 / 2, 1 / 2, 0, 0, 1 / 2, 0, 1 / 2])
        chosen = [0, 2]
        nonzero = [1, 2, 5, 7]
        nonzero_binary = [aux.decimal_to_binary(i, 3) for i in nonzero]

        W = create_w_from_binary(chosen, notchosen(chosen, 3), nonzero_binary).toarray()
        self.assertListEqual(W.tolist(), [[0, 1 / 2], [1 / 2, 0], [0, 0], [1 / 2, 1 / 2]])
        reduced_density_matrix = [[1 / 4, 0, 0, 1 / 4], [0, 1 / 4, 0, 1 / 4], [0, 0, 0, 0], [1 / 4, 1 / 4, 0, 1 / 2]]
        self.assertListEqual(W.dot(W.T).tolist(), reduced_density_matrix)

        chosen = [1, 2]
        W = create_w_from_binary(chosen, notchosen(chosen, 3), nonzero_binary).toarray()
        reduced_density_matrix = [[0, 0, 0, 0], [0, 1 / 2, 1 / 4, 1 / 4], [0, 1 / 4, 1 / 4, 0], [0, 1 / 4, 0, 1 / 4]]
        self.assertListEqual(W.dot(W.T).tolist(), reduced_density_matrix)
        qnotchosen = [3 - 1 - i for i in notchosen(chosen, 3)]
        qmatrix = partial_trace(Statevector(state), qnotchosen).data.real.astype(dtype=float,
                                                                                           copy=False)
        self.assertListEqual(reduced_density_matrix, qmatrix.tolist())

    def test_entropy(self):
        nonzero_decimal = [0, 3]
        bell_state_density = DensityMatrix([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
        qreduced = partial_trace(bell_state_density, [0])
        entropy = qentropy(qreduced)
        W = create_w_from_binary([1], [0], [aux.decimal_to_binary(i, 2) for i in nonzero_decimal])
        sing_values = numpysvd(W.toarray(), compute_uv=False, hermitian=False)
        myentr = - np.sum([(i ** 2) * 2 * np.log2(i) for i in sing_values if i > 1e-16])
        self.assertEqual(entropy, myentr)

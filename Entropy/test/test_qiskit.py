from unittest import TestCase

from states import *
from time import time

class Test(TestCase):

    def test_reduced_matrix(self):
        # psi=[0,1/2,1/2,0,0,1/2,0,1/2], chosen=[0,2] -> nonzeros=[1,2,5,7], nonzero_binary= ['001','010','101','111']
        # rows = ['01'-> 1, 0, 3, 3]          cols= [0,1,0,1]
        # W = 1 / 2 [ [0,1] , [1,0], [0,0], [1,1] ]
        state = np.array([0, 1 / 2, 1 / 2, 0, 0, 1 / 2, 0, 1 / 2])
        chosen = [0, 2]
        nonzero = [1, 2, 5, 7]
        nonzero_binary = [aux.decimal_to_binary(i, 3) for i in nonzero]

        W = bip.create_w_from_binary(chosen, bip.notchosen(chosen, 3), nonzero_binary)
        myreduced = W.dot(W.T).toarray().tolist()
        expected = [[1 / 4, 0, 0, 1 / 4], [0, 1 / 4, 0, 1 / 4], [0, 0, 0, 0], [1 / 4, 1 / 4, 0, 1 / 2]]
        self.assertListEqual(myreduced, expected)
        qreduced = qt.quantum_info.partial_trace(qt.quantum_info.Statevector(state), [1]).data. \
            real.astype(dtype=float, copy=False).tolist()
        self.assertListEqual(qreduced, expected)

        chosen = [1, 2]
        W = bip.create_w_from_binary(chosen, bip.notchosen(chosen, 3), nonzero_binary)
        myreduced = W.dot(W.T).toarray().tolist()
        qreduced = qt.quantum_info.partial_trace(qt.quantum_info.Statevector(state), [2]).data. \
            real.astype(dtype=float, copy=False).tolist()
        expected = [[0., 0., 0., 0.], [0., 1 / 2, 1 / 4, 1 / 4], [0., 1 / 4, 1 / 4, 0.], [0., 1 / 4, 0., 1 / 4]]
        self.assertListEqual(myreduced, expected)
        self.assertListEqual(qreduced, expected)

        sparse_state = bip.coo_matrix(([1 / 2, 1 / 2, 1 / 2, 1 / 2], (nonzero, [0, 0, 0, 0])), shape=(8, 1)).tocsr()
        chosen = [0, 1]
        W = matrix_from_state(sparse_state, chosen, bip.notchosen(chosen, 3))
        myreduced = W.dot(W.T).toarray().tolist()
        qreduced = qt.quantum_info.partial_trace(qt.quantum_info.Statevector(state), [0]).data. \
            real.astype(dtype=float, copy=False).tolist()
        expected = [[1 / 4, 0., 1 / 4, 1 / 4], [0., 1 / 4, 0., 0.], [1 / 4, 0., 1 / 4, 1 / 4],
                    [1 / 4, 0., 1 / 4, 1 / 4]]
        self.assertListEqual(myreduced, expected)
        self.assertListEqual(qreduced, expected)

    def test_W_coefficients_not_1(self):
        # psi = (2 001 + 010 + 101 ) / sqrt(6)
        norm = 1 / np.sqrt(6)
        data = [2 * norm, 1 * norm, 1 * norm]
        row = [1, 2, 5]
        col = [0, 0, 0]
        sparse_state = bip.coo_matrix((data, (row, col)), shape=(8, 1)).tocsr()
        psi = [0, 2 * norm, norm, 0, 0, norm, 0, 0]
        state = np.array(psi)
        chosen = [0, 1]
        notchosen = [2]
        W = matrix_from_state(sparse_state, chosen, notchosen)
        myreduced = W.dot(W.T).toarray().tolist()
        qreduced = qt.quantum_info.partial_trace(qt.quantum_info.Statevector(state), [0]).data. \
            real.astype(dtype=float, copy=False).tolist()
        print(myreduced)
        print(qreduced)
        self.assertTrue(False)

    def test_operator_not_doable(self):
        k = 10
        L = 5
        Y = 13
        N = 21
        nonzeros_decimal = aux.nonzeros_decimal(k, N, Y)
        state = construct_state(k, L, nonzeros_decimal)
        qstate = qt.quantum_info.Statevector(state.toarray())

        with self.assertRaises(MemoryError):
            applyIQFT(L, qstate).data.tolist()
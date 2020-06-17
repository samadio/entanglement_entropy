from unittest import TestCase

from src.states import *

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
        W = matrix_from_sparse_modular_state(sparse_state, chosen, bip.notchosen(chosen, 3))
        myreduced = W.dot(W.T).toarray().tolist()
        qreduced = qt.quantum_info.partial_trace(qt.quantum_info.Statevector(state), [0]).data. \
            real.astype(dtype=float, copy=False).tolist()
        expected = [[1 / 4, 0., 1 / 4, 1 / 4], [0., 1 / 4, 0., 0.], [1 / 4, 0., 1 / 4, 1 / 4],
                    [1 / 4, 0., 1 / 4, 1 / 4]]
        self.assertListEqual(myreduced, expected)
        self.assertListEqual(qreduced, expected)
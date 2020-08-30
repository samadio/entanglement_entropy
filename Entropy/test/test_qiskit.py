from unittest import TestCase

from states import *

class Test(TestCase):

    def test_reduced_matrix(self):
        # psi=[0,1/2,1/2,0,0,1/2,0,1/2], chosen=[0,2] -> nonzeros=[1,2,5,7], nonzero_binary= ['001','010','101','111']
        # rows = ['01'-> 1, 0, 3, 3]          cols= [0,1,0,1]
        # W = 1 / 2 [ [0,1] , [1,0], [0,0], [1,1] ]
        state = np.array([0, 1 / 2, 1 / 2, 0, 0, 1 / 2, 0, 1 / 2])
        chosen = [0, 2]
        notchosen = [1]
        nonzero = [1, 2, 5, 7]
        nonzero_binary = [aux.decimal_to_binary(i, 3) for i in nonzero]

        myreduced = density_matrix_from_state_dense(state, chosen, notchosen).tolist()
        expected = [[1 / 4, 0, 0, 1 / 4], [0, 1 / 4, 0, 1 / 4], [0, 0, 0, 0], [1 / 4, 1 / 4, 0, 1 / 2]]
        self.assertListEqual(myreduced, expected)
        qreduced = qt.quantum_info.partial_trace(qt.quantum_info.Statevector(state), notchosen).data. \
            real.astype(dtype=float, copy=False).tolist()
        self.assertListEqual(qreduced, expected)

        chosen = [0, 1]
        notchosen = [2]
        myreduced = density_matrix_from_state_dense(state, chosen, notchosen).tolist()
        qreduced = qt.quantum_info.partial_trace(qt.quantum_info.Statevector(state), notchosen).data. \
            real.astype(dtype=float, copy=False).tolist()
        expected = [[0., 0., 0., 0.], [0., 1 / 2, 1 / 4, 1 / 4], [0., 1 / 4, 1 / 4, 0.], [0., 1 / 4, 0., 1 / 4]]
        self.assertListEqual(myreduced, expected)
        self.assertListEqual(qreduced, expected)

        state = bip.coo_matrix(([1 / 2, 1 / 2, 1 / 2, 1 / 2], (nonzero, [0, 0, 0, 0])), shape=(8, 1)).toarray().flatten()
        chosen = [1, 2]
        notchosen = [0]
        myreduced = density_matrix_from_state_dense(state, chosen, notchosen).tolist()
        qreduced = qt.quantum_info.partial_trace(qt.quantum_info.Statevector(state), notchosen).data. \
            real.astype(dtype=float, copy=False).tolist()
        expected = [[1 / 4, 0., 1 / 4, 1 / 4], [0., 1 / 4, 0., 0.], [1 / 4, 0., 1 / 4, 1 / 4],
                    [1 / 4, 0., 1 / 4, 1 / 4]]
        self.assertListEqual(myreduced, expected)
        self.assertListEqual(qreduced, expected)
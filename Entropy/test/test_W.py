from states import *
from unittest import TestCase

class Test(TestCase):

    def test_reduced_matrix(self):
        state = np.array([0, 1 / 2, 1 / 2, 0, 0, 1 / 2, 0, 1 / 2])
        nonzero = [1, 2, 5, 7]
        for chosen in [[0], [1], [2]]:
            myreduced = density_matrix_from_state_dense(state, chosen, bip.notchosen(chosen, 3))
            qreduced = qt.quantum_info.partial_trace(qt.quantum_info.Statevector(state), bip.notchosen(chosen, 3)).data. \
                real.astype(dtype=float, copy=False)
            np.testing.assert_array_almost_equal(myreduced, qreduced)
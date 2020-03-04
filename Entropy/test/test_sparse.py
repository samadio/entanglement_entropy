from unittest import TestCase

from states import *


class Test(TestCase):
    def test_sparse(self):
        k = 7
        Y = 13
        N = 21
        L = 5
        nonzero_decimal = aux.nonzeros_decimal(k, N, Y)
        state = construct_state(k, L, nonzero_decimal)
        tries = 50
        for i in range(tries):
            chosen = bip.random_bipartition(range(k + L), (k + L) // 2)
            notchosen = bip.notchosen(chosen, k + L)
            W_sparse = matrix_from_state(state, chosen, notchosen, True).toarray()
            W_dense = matrix_from_state(state, chosen, notchosen, False)
            self.assertListEqual(W_sparse.tolist(), W_dense.tolist())

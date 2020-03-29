from unittest import TestCase

from src.states import *
from time import time


class Test(TestCase):
    def test_svd_ordered(self):
        N = 129
        Y = 22
        L = aux.lfy(N)
        k = 2 * L
        nonzero = aux.nonzeros_decimal(k, Y, N)

        state = construct_modular_state(k, L, nonzero)
        chosen = bip.random_bipartition(range(k + L), (k + L) // 2)
        W = matrix_from_state_modular(state, chosen, bip.notchosen(chosen, k + L), False)

        print("scipy sparse svd")
        whole = time()
        sv = scipy.sparse.linalg.svds(W, k=min(W.shape) - 2, return_singular_vectors=False)
        whole = time() - whole
        print(whole)

        W = matrix_from_state_modular(state, chosen, bip.notchosen(chosen, k + L), False)
        print("numpy total svd")
        numpy_whole = time()
        num_sv = numpysvd(W, compute_uv=False)
        numpy_whole = time() - numpy_whole
        print(numpy_whole)

        np.testing.assert_array_almost_equal(-np.sort(-num_sv)[:len(sv)], -np.sort(-sv))

        def round_local(x, W):
            return aux.math.ceil(x * (min(W.shape) - 2))

        print("time scaling for computing scipy sparse truncated svd")
        reduced_times = []
        for x in np.linspace(0.1, 1.0, num=10):
            red = time()
            number_of_sv = round_local(x, W)
            reduced_sv = scipy.sparse.linalg.svds(W, which="LM", k=number_of_sv, return_singular_vectors=False)
            red = time() - red
            reduced_times.append(red)

            np.testing.assert_array_almost_equal(-np.sort(-sv)[:len(reduced_sv)], -np.sort(-reduced_sv))

        print(100 * np.array(reduced_times) / whole)

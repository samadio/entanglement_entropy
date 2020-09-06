from src.states import *
from time import time
from notify_run import Notify

with open("Svd_convergence_scaling.txt", "a+") as file:
    program_time = time()

    N = 129
    Y = 16
    L = aux.lfy(N)
    k = 2 * L
    nonzero = aux.nonzeros_decimal(k, Y, N)

    state = construct_modular_state(k, L, nonzero)
    chosen = bip.random_bipartition(range(k + L), (k + L) // 2)
    W = matrix_from_sparse_modular_state(state, chosen, bip.notchosen(chosen, k + L), False)
    percentages = np.linspace(0.02, 0.2, num=10)

    print("L = " + str(L) + ", k = " + str(k), file=file)

    print("scipy sparse total svd time", file=file)
    whole = time()
    sv = scipy.sparse.linalg.svds(W, k=min(W.shape) - 2, return_singular_vectors=False)
    whole = time() - whole
    print(whole, file=file)

    W = matrix_from_sparse_modular_state(state, chosen, bip.notchosen(chosen, k + L), False)
    print("\nnumpy total svd time", file=file)
    numpy_whole = time()
    num_sv = numpysvd(W, compute_uv=False)
    numpy_whole = time() - numpy_whole
    print(numpy_whole, file=file)

    np.testing.assert_array_almost_equal(-np.sort(-num_sv)[:len(sv)], -np.sort(-sv))

    W = matrix_from_sparse_modular_state(state, chosen, bip.notchosen(chosen, k + L), False)
    print("\nscipy total svd time", file=file)
    scipy_whole = time()
    _ = scipy.linalg.svd(W, compute_uv=False, overwrite_a=False, check_finite=False)
    scipy_whole = time() - scipy_whole
    print(scipy_whole, file=file)


    def round_local(x, W):
        return aux.math.ceil(x * (min(W.shape) - 2))


    print("time scaling for computing scipy sparse truncated svd\n", file=file)
    reduced_times = []
    W = matrix_from_sparse_modular_state(state, chosen, bip.notchosen(chosen, k + L), False)
    for x in percentages:
        red = time()
        number_of_sv = round_local(x, W)
        reduced_sv = scipy.sparse.linalg.svds(W, which="LM", k=number_of_sv, return_singular_vectors=False)
        red = time() - red
        reduced_times.append(red)

        np.testing.assert_array_almost_equal(-np.sort(-sv)[:len(reduced_sv)], -np.sort(-reduced_sv))

    print("100 * time for computing x*total singular values /  total sparse time:  for x in linspace(0.2,0.02, num=10)",
          file=file)
    print(100 * np.array(reduced_times) / whole, file=file)

    print("\ntime for x*sv / scipy_time", file=file)
    print(np.array(reduced_times) / scipy_whole, file=file)

    program_time = (time() - program_time) / 60

    notice = Notify()
    notice.send("Work done after " + str(program_time) + " minutes")
    print("Work done after " + str(program_time) + " minutes", file=file)

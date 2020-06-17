from math import ceil
from src.states import *
from notify_run import Notify
from time import time


program_time = time()

sparse = False
Y = 13
numbers = [15, 21, 33, 66, 129, 300]
L_list = [aux.lfy(N) for N in numbers]

number_of_bipartitions = 20
sampled_bipartitions = [[[bip.random_bipartition(range(k + L), (k + L) // 2) for j in range(number_of_bipartitions)] \
                         for k in range(1, 2 * L + 1)] \
                        for L in L_list]
files = [open("svd_convergence/" + "Partition " + str(j) + ".txt", 'a+') for j in range(number_of_bipartitions)]

for i, N in enumerate(numbers):
    L = L_list[i]
    [print("\nL = " + str(L) + "\n", file=file) for file in files]
    [print("[", file=file) for file in files]

    nonzeros = [aux.nonzeros_decimal(2 * aux.lfy(N), Y, N) for N in numbers]
    for k in range(1, 2 * L + 1):
        notchosen_list = [bip.notchosen(partition, k + L) for partition in sampled_bipartitions[i][k - 1]]
        current_state = construct_modular_state(k, L, nonzeros[i][: 2 ** k])
        for j in range(number_of_bipartitions):
            W = matrix_from_sparse_modular_state(current_state, sampled_bipartitions[i][k - 1][j], notchosen_list[j], sparse)
            svd = scipy.linalg.svd(W, compute_uv=False, overwrite_a=True, check_finite=False)
            svd = svd ** 2
            svd = - np.sort(-svd)
            svd[np.abs(svd) < 2e-16] = 1

            entropy = [- np.sum(svd[: ceil(2 * upper * len(svd))] * np.log2(svd[:ceil(2 * upper * len(svd))])) for upper in
                       [.01, .02, .03, .04, .05, .06, .07, .08, .09, 0.1, .5]]

            print(entropy, file=files[j], end=",")

    current_state = apply_IQFT(L, current_state)
    for j in range(number_of_bipartitions):
        W = matrix_from_dense_state_IQFT(current_state, sampled_bipartitions[i][k - 1][j], notchosen_list[j])
        svd = scipy.linalg.svd(W, compute_uv=False, overwrite_a=True, check_finite=False)
        svd = svd ** 2
        svd = - np.sort(-svd)

        # so that eigenvalues below threshold do not contribute to sum
        svd[np.abs(svd) < 2e-16] = 1

        entropy = [- np.sum(svd[:ceil(2 * upper * len(svd))] * np.log2(svd[:ceil(2 * upper * len(svd))])) for upper in
                   [.01, .02, .03, .04, .05, .06, .07, .08, .09, 0.1, .5]]

        print(entropy, file=files[j])
        print("]", file=files[j])

program_time = (time() - program_time) / 60

notice = Notify()
notice.send("Work done after " + str(program_time) + " minutes")
print("Work done after " + str(program_time) + " minutes", file=files[0])
[file.close() for file in files]

from src.states import *

sparse = False
Y = 13
numbers = [15, 21, 33, 66, 129]
L_list = [aux.lfy(N) for N in numbers]

number_of_bipartitions = 20
sampled_bipartitions = [[[bip.random_bipartition(range(k + L), (k + L) // 2) for j in range(number_of_bipartitions)] \
                         for k in range(1, 2 * L + 1)] \
                        for L in L_list]

with open("svd_convergence.txt", "a+") as file:

    for i, N in enumerate(numbers):
        L = L_list[i]
        print("L = " + str(L), file=file)
        print("")
        nonzeros = [aux.nonzeros_decimal(2 * aux.lfy(N), Y, N) for N in numbers]
        for k in range(1, 2 * L + 1):
            notchosen_list = [bip.notchosen(partition, k + L) for partition in sampled_bipartitions[i][k - 1]]
            current_state = construct_modular_state(k, L, nonzeros[i][: 2 ** k])
            print("k = " + str(k) + " : S with fraction of singular values: [0.1,0.2,..,1]. Last one missing", file=file)
            for j in range(number_of_bipartitions):
                for x in np.linspace(0.1, 1.0, num=10):
                    W = matrix_from_sparse_modular_state(current_state, sampled_bipartitions[i][k - 1][j], notchosen_list[j])
                    svd = scipy.sparse.linalg.svds(W, which="LM",
                                                   k=min(aux.math.ceil(x * (min(W.shape) - 1)), (min(W.shape) - 2)),
                                                         return_singular_vectors=False)
                    svd = svd ** 2
                    entropy = - np.sum([i * np.log2(i) for i in svd if i > 1e-17])
                    print(entropy, file=file, end=', ')
                print("\n ", file=file)

        print("IQFT:", file=file)
        current_state = apply_IQFT(L, current_state)
        for j in range(number_of_bipartitions):
            for x in np.linspace(0.1, 1.0, num=10):
                W = matrix_from_dense_state_IQFT(current_state, sampled_bipartitions[i][k - 1][j], notchosen_list[j])
                svd = scipy.sparse.linalg.svds(W, which="LM",
                                               k=min(aux.math.ceil(x * (min(W.shape) - 1)), (min(W.shape) - 2)),
                                               return_singular_vectors=False)
                svd = svd ** 2
                entropy = - np.sum([i * np.log2(i) for i in svd if i > 1e-17])
                print(entropy, file=file, end=', ')
            print("\n ", file=file)
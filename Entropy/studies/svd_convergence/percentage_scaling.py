from src.states import *
from time import time

with open("Svd_2_percent_time_scaling.txt", "a+") as file:
    for i in range(1):
        ratios = []
        for N in [21, 33, 65, 129, 301]:
            Y = 16
            L = aux.lfy(N)
            k = 2 * L
            nonzero = aux.nonzeros_decimal(k, Y, N)

            state = construct_modular_state(k, L, nonzero)
            chosen = bip.random_bipartition(range(k + L), (k + L) // 2)
            W = matrix_from_state_modular(state, chosen, bip.notchosen(chosen, k + L), False)

            #print("L = " + str(L) + ", k = " + str(k), file=file)

            W = matrix_from_state_modular(state, chosen, bip.notchosen(chosen, k + L), False)
            #print("\nnumpy total svd time", file=file)
            numpy_whole = time()
            num_sv = numpysvd(W, compute_uv=False)
            numpy_whole = time() - numpy_whole
            #print(numpy_whole, file=file)


            def round_local(x, W):
                return aux.math.ceil(x * min(W.shape))

            W = matrix_from_state_modular(state, chosen, bip.notchosen(chosen, k + L), False)
            red = time()
            number_of_sv = round_local(0.02, W)
            print(number_of_sv)
            reduced_sv = scipy.sparse.linalg.svds(W, which="LM", k=number_of_sv, return_singular_vectors=False)
            red = time() - red
            ratios.append(red / numpy_whole)

        #print("Ratios 2 percent time / numpy time", file=file)
        print(ratios, file=file)
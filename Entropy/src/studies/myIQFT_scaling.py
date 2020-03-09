from states import *
from time import time

with open("computing IQFT.txt", "a+") as file:
    numbers = [15, 22, 35, 67, 129]
    for N in numbers:
        L = aux.lfy(N)
        Y = 13
        k = 2 * L
        row = 6

        print("nqubits = " + str(3 * L), file=file)
        start_time = time()
        nonzeros = aux.nonzeros_decimal(k, N, Y)
        state = construct_modular_state(k, L, nonzeros)
        print("    Inizialization time: " + str(time() - start_time), file=file)

        start_time = time()
        ide = identity(2 ** L)
        IQFT_row = operator_IQFT_row(2 * L, row)
        IQFT_operator_global = tensor(IQFT_row, ide)
        print("    Creating global operator for single row time: " + str(time() - start_time), file=file)

        start_time = time()
        np.dot(IQFT_operator_global, state)

        print("    dot product time: " + str(time() - start_time), file=file)


## from dotproduct to sum

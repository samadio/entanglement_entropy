from src.states import *
from time import time
from qiskit.aqua.algorithms import Shor

with open("computing IQFT.txt", "a+") as file:
    numbers = [500]#[15, 27, 35, 67, 129]  # , 261]
    for N in numbers:
        L = aux.lfy(N)
        Y = 13
        k = 2 * L
        row = 6

        print("nqubits = " + str(3 * L), file=file)
        nonzeros = aux.nonzeros_decimal(k, N, Y)
        state = construct_modular_state(k, L, nonzeros)

        print("state constructed")

        start_time = time()
        '''
        ide = identity(2 ** L)
        IQFT_op = coo_matrix(operator_IQFT(2 * L))
        IQFT_operator_global = tensor(ide, IQFT_op, format='csr')
        print("    Creating global operator time: " + str(time() - start_time), file=file)

        start_time = time()
        test2 = IQFT_operator_global.dot(state)

        start_time = time()
        nonzeros = np.array(nonzeros, dtype=np.int64)
        state = state.toarray()
        IQFT_row = operator_IQFT_row(2 * L, row)
        test1 = []
        for i in range(2 ** L):
            relevant = nonzeros[(nonzeros % ((2 ** L) + i)) == 0] / (i + 2 ** L)
            #relevant is empty
            #if max(relevant.shape) == 0:
            #    test1.append(0)
            #    continue
            test1.append(np.sum(IQFT_row[relevant.astype(np.int64)]) * len(nonzeros) ** (- 1 / 2))
        np.testing.assert_array_almost_equal(test1, test2.toarray().reshape((2 ** L,)), decimal=12)
        print(str(N))'''

        final_state = apply_IQFT(L, state)

        #backend = qt.Aer.get_backend('statevector_simulator')
        #shor_circ = Shor(N, Y).run(backend)

        print("    L=9 circuit time: " + str(time() - start_time), file=file)

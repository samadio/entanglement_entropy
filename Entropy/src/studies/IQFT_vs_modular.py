# How much IQFT takes more time than Modular exponentiation? Using both scipy,numpy and qiskit?
import qiskit.quantum_info as qinfo
from time import time
from auxiliary.auxiliary import lfy, decimal_to_binary
from auxiliary.bipartitions import random_bipartition
import states as st

for sparse in [True, False]:

    if sparse:
        string = "sparse"
    else:
        string = "numpy"

    with open("IQFT_vs_modular_" + string + ".txt", "a+") as file:

        #file = open("IQFT_vs_modular_" + string, "a+")#use the fucking context managers

        Y = 13
        numbers = [15, 21, 33, 66]
        L_list = [lfy(N) for N in numbers]
        number_of_bipartitions = 100
        print("Number of bipartitions: " + str(number_of_bipartitions), file=file)

        nonzeros_decimal = [[m * 2 ** lfy(N) + ((Y ** m) % N) for m in range(2 ** (2 * lfy(N)))] for N in numbers]

        sampled_bipartitions = [[[random_bipartition(range(k + L), (k + L) // 2) for j in range(number_of_bipartitions)] \
                                 for k in range(1, 2 * L + 1)] \
                                for L in L_list]

        print("qiskit:", file=file)
        for i in range(len(numbers)):
            N = numbers[i]
            L = L_list[i]

            modular_time = time()

            for k in range(1, 2 * L + 1):
                current_state = qinfo.Statevector(st.construct_state(k, L, nonzeros_decimal[i][: 2 ** k]).toarray())

                results = [qinfo.entropy(qinfo.partial_trace(current_state, chosen)) for chosen in
                           sampled_bipartitions[i][k - 1]]

            modular_time = time() - modular_time

            IQFT_time = time()

            final_state = st.applyIQFT_circuit(L, st.construct_state(2 * L, L, nonzeros_decimal[i]))
            results = [qinfo.entropy(qinfo.partial_trace(final_state, chosen)) for chosen in sampled_bipartitions[i][2 * L - 1]]
            IQFT_time = time() - IQFT_time

            print("      N =" + str(N) + ", Y = " + str(Y) + ", qubits = " + str(3 * L), file=file)
            print("           modular time: " + str(modular_time), file=file)
            print("           IQFT time: " + str(IQFT_time), file=file)

        print(sparse, file=file)
        for i in range(len(numbers)):
            N = numbers[i]
            L = L_list[i]

            modular_time = time()
            for k in range(1, 2 * L + 1):
                current_state = st.construct_state(k, L, nonzeros_decimal[i][: 2 ** k])
                results = [st.entanglement_entropy_from_state(current_state, chosen, sparse) for chosen in sampled_bipartitions[i][k - 1]]

            modular_time = time() - modular_time

            IQFT_time = time()

            final_state = st.applyIQFT_circuit(L, st.construct_state(2 * L, L, nonzeros_decimal[i]))

            results = [qinfo.entropy(qinfo.partial_trace(final_state, chosen)) for chosen in sampled_bipartitions[i][2 * L - 1]]

            IQFT_time = time() - IQFT_time

            print("      N =" + str(N) + ", Y = " + str(Y) + ", qubits = " + str(3 * L), file=file)
            print("           modular time: " + str(modular_time), file=file)
            print("           IQFT time: " + str(IQFT_time), file=file)

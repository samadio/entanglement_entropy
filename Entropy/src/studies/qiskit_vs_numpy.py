from time import time as time
from qiskit import quantum_info as qinfo

from auxiliary.auxiliary import lfy, decimal_to_binary
import numpy as np
import states as st
from auxiliary.bipartitions import random_bipartition, notchosen, sparsesvd, entropy

elenco = [15, 21]  # , 66, 129, 260]

elenco_L = [lfy(i) for i in elenco]
print("Y=13, N:" + str([i for i in elenco]))

Y = 13
qtime = []
mytime = []

number_of_bipartitions = 100
print("Number of bipartitions: " + str(number_of_bipartitions))

sampled_bipartitions = [[[random_bipartition(range(k + L), (k + L) // 2) for j in range(number_of_bipartitions)] \
                         for k in range(1, 2 * L + 1)] \
                        for L in elenco_L]

nonzeros_decimal = [[m * 2 ** lfy(N) + ((Y ** m) % N) for m in range(2 ** (2 * lfy(N)))] for N in elenco]

print("Starting qiskit")

total_entropy = []

for i in range(len(elenco)):
    entropy_fixed_N = []

    qiskit_time = time()

    N = elenco[i]
    L = elenco_L[i]

    for k in range(1, 2 * L + 1):
        considered_qubits = range(k + L)
        bipartition_size = (k + L) // 2

        current_state = qinfo.Statevector(st.construct_state(k, L, nonzeros_decimal[i][: 2 ** k]).toarray())

        entropy_fixed_N.append(
            [qinfo.entropy(qinfo.partial_trace(current_state, chosen)) for chosen in sampled_bipartitions[i][k - 1]])

    final_state = st.applyIQFT_circuit(L, st.construct_state(2 * L, L, nonzeros_decimal[i]))
    entropy_fixed_N.append(
        [qinfo.entropy(qinfo.partial_trace(final_state, chosen)) for chosen in sampled_bipartitions[i][2 * L - 1]])
    qtime.append(time() - qiskit_time)
    total_entropy.append(entropy_fixed_N)
    print("Finished N=" + str(N))

print("Starting scipy")

total_results = []

for i in range(len(elenco)):
    results_fixed_N = []
    scipy_time = time()

    N = elenco[i]
    L = elenco_L[i]

    for k in range(1, 2 * L + 1):
        considered_qubits = range(k + L)
        bipartition_size = (k + L) // 2

        nonzero_binary = [decimal_to_binary(i, k + L) for i in nonzeros_decimal[i][:2 ** k]]

        results_fixed_N.append([entropy(k, L, chosen, nonzero_binary) for chosen in sampled_bipartitions[i][k - 1]])

    final_state = st.applyIQFT_circuit(L, st.construct_state(2 * L, L, nonzeros_decimal[i]))

    results_fixed_N.append(
        [qinfo.entropy(qinfo.partial_trace(final_state, chosen)) for chosen in sampled_bipartitions[i][2 * L - 1]])

    mytime.append(time() - scipy_time)
    total_results.append(results_fixed_N)
    print("Finished N=" + str(N))

diff = []

for i in range(len(elenco)):
    diff.append(np.max(np.abs(np.array(total_results[i]) - np.array(total_entropy[i]))))

print("\nmaximum entropy difference for each L:")
print([i for i in diff])

print("\nqiskit times:")
print([i for i in qtime])
print("\nmy times: ")
print([i for i in mytime])

from entropies import *
from states import *
from time import time
from qiskit.quantum_info import partial_trace, entropy, Statevector


files = [open("qiskit_mycode.py", 'a+')]

N_list = [21, 33, 71, 132]

bipartitions_tested = 500
print("#Watching only states after IQFT\n#Number of bipartitions tested: " + str(bipartitions_tested), file=files[0])

first = True

for N in N_list:
    total_start = time()
    my_time = []
    qiskit_time = []
    L = aux.lfy(N)
    k = 2 * L
    Y = aux.coprime(N)
    or_state = construct_modular_state(k, L, aux.nonzeros_decimal(k, Y, N)).toarray().flatten()

    start = time()
    or_state = apply_IQFT(L, or_state)
    print("#IQFT time: ", time() - start, file=files[0])

    for i in range(bipartitions_tested):
        chosen = bip.random_bipartition(range(k + L), (k + L) // 2)
        notchosen = bip.notchosen(chosen, k + L)

        start = time()
        myentr = entanglement_entropy_from_state(or_state, chosen, sparse=False, gpu=False)
        my_time.append(time() - start)

        start = time()
        qreduced = entropy(partial_trace(Statevector(or_state), notchosen))
        qiskit_time.append(time() - start)
        np.testing.assert_almost_equal(qreduced, myentr, decimal=10)

    total_end = time()
    if first:
        print("my_time_" + str(L)+" =", np.mean(my_time[1:]), file=files[0])
    else:
        print("my_time_" + str(L)+" =", np.mean(my_time), file=files[0])
        first = False

    print("qiskit_time_" + str(L)+" =", np.mean(qiskit_time), file=files[0])

    print("total_time_" + str(L)+" =", total_end-total_start, file=files[0])

files[0].close()

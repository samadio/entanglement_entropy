from time import time
from states import *
from entropies import *

total_start = time()

numbers = [23, 35, 68,132]#[23, 35, 68, 132]
L_list = [aux.lfy(N) for N in numbers]
Y = 13
current_state = 0
number_of_bip = 1_000
means = [np.ones(2 * L + 1, dtype=float) * 512.32 for L in L_list]
stds = [np.ones(2 * L + 1, dtype=float) * 512.32 for L in L_list]

start = time()
for i, N in enumerate(numbers):
    flag = False
    L = L_list[i]
    for k in range(1, 2 * L + 1):
        current_state = construct_modular_state(k, L, aux.nonzeros_decimal(k, Y, N)).toarray().flatten()
        number_qubits = k + L
        bipartitions = [bip.random_bipartition(range(number_qubits), number_qubits // 2) for i in range(number_of_bip)]
        times = []
        for chosen in bipartitions:
            local_start = time()
            entropy = entanglement_entropy_from_state(current_state, chosen, sparse=False, gpu=True)
            if flag: times.append(time() - local_start)
            else: flag = True

        means[i][k - 1] = np.mean(times)
        stds[i][k - 1] = np.sqrt(np.var(times))

    current_state = apply_IQFT(L, current_state)
    bipartitions = [bip.random_bipartition(range(number_qubits), number_qubits // 2) for i in range(number_of_bip)]
    times = []
    for chosen in bipartitions:
        local_start = time()
        entropy = entanglement_entropy_from_state(current_state, chosen, sparse=False, gpu=True)
        if flag: times.append(time() - local_start)
        else: flag = True

    means[i][-1] = np.mean(times)
    stds[i][ 1] = np.sqrt(np.var(times))


total_time = time() - start

files = [open("/home/samadio/entropy/Entropy/test_times_L_" + str(L) + ".py", 'a+') for L in L_list]

[print("#"+str(number_of_bip) + " bipartitions, PARALLEL COMPILING \new_means = " + str(means[i].tolist()), file=files[i]) for i in range(len(numbers))]
[print("new_stds = " + str(stds[i].tolist()), file=files[i]) for i in range(len(numbers))]
print("#total time " + str(total_time/60) + " mins", file=files[0])
[file.close() for file in files]

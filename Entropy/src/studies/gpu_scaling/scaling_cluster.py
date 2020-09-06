from time import time
from states import *
from entropies import *

import cupy as cp
from cupy.linalg import eigvalsh as gpu_eigh


total_start = time()

numbers = [23, 35, 68,132]#[23, 35, 68, 132]
L_list = [aux.lfy(N) for N in numbers]
Y = 13
current_state = 0
number_of_bip = 1_000
constr_means = [np.ones(2 * L + 1, dtype=float) * 512.32 for L in L_list]
constr_stds = [np.ones(2 * L + 1, dtype=float) * 512.32 for L in L_list]
eigh_means = [np.ones(2 * L + 1, dtype=float) * 512.32 for L in L_list]
eigh_stds = [np.ones(2 * L + 1, dtype=float) * 512.32 for L in L_list]

start = time()
for i, N in enumerate(numbers):
    flag = False
    L = L_list[i]
    nonzeros = aux.nonzeros_decimal(2 * L, Y, N)
    for k in range(1, 2 * L + 1):
        current_state = construct_modular_state(k, L, nonzeros[:2**k]).toarray().flatten()
        number_qubits = k + L
        bipartitions = [bip.random_bipartition(range(number_qubits), number_qubits // 2) for i in range(number_of_bip)]
        constr_times = []
        eigh_times = []
        for chosen in bipartitions:
            local_start = time()
            notchosen = bip.notchosen(chosen, number_qubits)
            W = W_from_state_numba(current_state, chosen, notchosen)
            if flag: constr_times.append(time() - local_start)
            else: pass
            local_start = time()
            W = cp.array(W)
            rho = W.dot(W.conj().T)
            eig = gpu_eigh(rho)
            eig = eig[eig > 1e-5]
            a = cp.log2(eig)
            entr = cp.asnumpy(- cp.sum(eig * a))
            if flag: eigh_times.append(time() - local_start)
            else: flag = True       

        constr_means[i][k - 1] = np.mean(constr_times)
        constr_stds[i][k - 1] = np.sqrt(np.var(constr_times))
        eigh_means[i][k - 1] = np.mean(eigh_times)
        eigh_stds[i][k - 1] = np.sqrt(np.var(eigh_times))

    current_state = apply_IQFT(L, current_state)
    bipartitions = [bip.random_bipartition(range(number_qubits), number_qubits // 2) for i in range(number_of_bip)]
    constr_times = []
    eigh_times = []
    for chosen in bipartitions:
        local_start = time()
       	notchosen = bip.notchosen(chosen, number_qubits)                    
        W = W_from_state_numba(current_state, chosen, notchosen)
        if flag: constr_times.append(time() - local_start)
        else: pass       
       	local_start = time()
       	W = cp.array(W)
        rho = W.dot(W.conj().T)
        eig = gpu_eigh(rho)
        eig = eig[eig > 1e-5]
        a = cp.log2(eig)
        entr = cp.asnumpy(- cp.sum(eig * a))
        if flag: eigh_times.append(time() - local_start)
        else: flag = True 

    constr_means[i][- 1] = np.mean(constr_times)
    constr_stds[i][- 1] = np.sqrt(np.var(constr_times))
    eigh_means[i][- 1] = np.mean(eigh_times)
    eigh_stds[i][- 1] = np.sqrt(np.var(eigh_times))


total_time = time() - start

files = [open("/home/samadio/entropy/Entropy/constr_vs_eig_times_L_" + str(L) + ".py", 'a+') for L in L_list]

[print("#"+str(number_of_bip) + " bipartitions, PARALLEL COMPILING \nconstr_means = " + str(constr_means[i].tolist())+ "\neigh_means = " + str(eigh_means[i].tolist()), file=files[i]) for i in range(len(numbers))]
[print("constr_stds = " + str(constr_stds[i].tolist())+"\neigh_stds = " + str(eigh_stds[i].tolist()), file=files[i]) for i in range(len(numbers))]
print("#total time " + str(total_time/60) + " mins", file=files[0])
[file.close() for file in files]

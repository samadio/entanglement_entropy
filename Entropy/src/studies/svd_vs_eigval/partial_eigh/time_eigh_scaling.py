from auxiliary import auxiliary as aux
from auxiliary import bipartitions as bip
from states import *

from time import time
import matplotlib.pyplot as plt
from math import ceil
from jax.scipy.linalg import eigh as jseigh


Y = 13
numb = [21, 33]#, 66, 129]
L_list = []
tries = 20
percentages = [5,10,15,20,25,30,100]
L_times = np.zeros((tries, len(percentages)))
means = np.zeros((len(numb), len(percentages)))

with open("jax_eigh_time_scaling.txt", "a+") as file:
    for idx, N in enumerate(numb):
        L = aux.lfy(N)
        L_list.append(L)
        k = 2 * L
        number_of_qubits = k + L
        state = construct_modular_state(k, L, aux.nonzeros_decimal(k, Y, N))
        chosens = [bip.random_bipartition(range(number_of_qubits), number_of_qubits // 2) for i in range(tries)]
        for i, perc in enumerate(percentages):
            number_total_eigvals = 2 ** (number_of_qubits // 2)
            number_eigvals_computed = int(ceil(perc * number_total_eigvals / 100))
            for j in range(tries):
                chosen = chosens[i]
                notchosen = bip.notchosen(chosen, k + L)
                W = matrix_from_sparse_modular_state(state, chosen, notchosen, False)

                start = time()
                sl = jseigh(W.dot(W.T), eigvals_only=True, turbo=False, check_finite=False, overwrite_a=True,
                            overwrite_b=True, eigvals=(number_total_eigvals - 1 - number_eigvals_computed,number_total_eigvals - 1))
                L_times[j, i] = time() - start
        mean = np.mean(L_times, axis=0)
        means[idx] = mean

    print(means, file=file)

for i in range(len(numb)):
    L = L_list[i]
    x = range(1, len(percentages) + 1)
    plt.plot(x, means[i], c='blue', marker='-o')

    plt.xticks(list(x))
    plt.xlabel("percentage of eigenvalues")
    plt.ylabel("seconds")
    plt.title("L = " + str(L))
    plt.savefig("partial_eigvals_scaling_L=" + str(L) + ".png", dpi=200)
    plt.show()
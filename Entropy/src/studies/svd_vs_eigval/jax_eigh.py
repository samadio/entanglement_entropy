import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

command = 'export MKL_NUM_THREADS=3\nexport NUMEXPR_NUM_THREADS=3\nexport OMP_NUM_THREADS=3'
os.system(command)


from src.auxiliary.bipartitions import random_bipartition
from src.states import *

from time import time
import matplotlib.pyplot as plt
from jax.numpy.linalg import eigvalsh as jeigh


Y = 13
numb = [21, 33, 66, 129]
jax_time = []
L_list = []
tries = 100

with open("entropy_test.txt", "a+") as file:
    for N in numb:
        L = aux.lfy(N)
        k = 2 * L
        number_of_qubits = k + L
        state = construct_modular_state(k, L, aux.nonzeros_decimal(k, Y, N))
        chosens = [random_bipartition(range(number_of_qubits), number_of_qubits // 2) for i in range(tries)]
        for i in range(tries):
            L_list.append(L)
            chosen = chosens[i]
            notchosen = bip.notchosen(chosen, k + L)
            W = matrix_from_state_modular(state, chosen, notchosen, False)

            start = time()

            l = jeigh(W.dot(W.T))
            jax_time.append(time() - start)
            l = np.array(l)
            l[l < 1e-15] = 1
            entropy = np.sum(l * np.log2(l))
            print(entropy, file=file)

    numpy_means = [0.0014, 0.05, 0.4, 14.6]
    numpy_std = [0.0006, 0.006, 0.06, 0.27]
    for i in range(len(numb)):
        L = aux.lfy(numb[i])

        x = range(1, tries+1)
        plt.plot(x, numpy_means[i] * np.ones(100), 'k-', label="numpy svd")
        plt.fill_between(x, numpy_means[i] - 3 * numpy_std[i], numpy_means[i] + 3 * numpy_std[i], alpha=0.3)
        plt.scatter(range(1, tries + 1), jax_time[i * tries:(i + 1) * tries], c='red', marker='o',
                    label="jax svd")

        plt.legend()
        plt.xticks(range(0, tries+10, 10))
        plt.xlabel("tries")
        plt.ylabel("seconds")
        plt.title("Methods comparison, L = " + str(L))
        plt.savefig("numpy_vs_jax_L_" + str(L) + ".png", dpi=200)
        plt.show()

        mean = np.mean(jax_time[i * tries:(i + 1) * tries])
        sv_var = np.var(jax_time[i * tries:(i + 1) * tries])

        print("JSvd time, L = " + str(L))
        print(str(mean) + " ± " + str(np.sqrt(sv_var)))
        print("")

'''

'''
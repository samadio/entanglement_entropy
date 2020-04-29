'''import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

command = 'export MKL_NUM_THREADS=3\nexport NUMEXPR_NUM_THREADS=3\nexport OMP_NUM_THREADS=3'
os.system(command)
'''

from src.auxiliary import auxiliary as aux
from src.auxiliary import bipartitions as bip
from src.states import *

from time import time
import matplotlib.pyplot as plt
from jax.numpy.linalg import eigvalsh as jeigh
from numpy.linalg import eigvalsh as eigh

Y = 13
numb = [21, 33, 66, 129]#, 357]
jax_time = []
numpy_time = []
L_list = []
tries = 100

with open("entropy_test.txt", "a+") as file:
    for N in numb:
        L = aux.lfy(N)
        k = 2 * L
        number_of_qubits = k + L
        state = construct_modular_state(k, L, aux.nonzeros_decimal(k, Y, N))
        chosens = [bip.random_bipartition(range(number_of_qubits), number_of_qubits // 2) for i in range(tries)]
        for i in range(tries):
            L_list.append(L)
            chosen = chosens[i]
            notchosen = bip.notchosen(chosen, k + L)
            W = matrix_from_state_modular(state, chosen, notchosen, False)

            if L == 9:
                print("look!")

            start = time()

            l = jeigh(W.dot(W.T))
            jax_time.append(time() - start)
            if L == 9:
                print("dont look!")

            l = np.array(l)
            l[l < 1e-15] = 1
            entropy = np.sum(l * np.log2(l))
            print(entropy, file=file)

            start = time()

            nl = eigh(W.dot(W.T))
            numpy_time.append(time() - start)
            nl[nl < 1e-15] = 1
            entropy = np.sum(nl * np.log2(nl))
            print(entropy, file=file)

means = []
std_dev = []
L_list = []
with open("jax_eigh.txt", "a+") as file:
    for i in range(len(numb)):
        L = aux.lfy(numb[i])
        L_list.append(L)
        x = range(1, tries + 1)
        plt.scatter(range(1, tries + 1), numpy_time[i * tries:(i + 1) * tries], c='blue', marker='o',
                    label="numpy eigvalsh")
        plt.scatter(range(1, tries + 1), jax_time[i * tries:(i + 1) * tries], c='green', marker='o',
                    label="jax eigvalsh")

        plt.legend()
        plt.xticks(range(0, tries + 10, 10))
        plt.xlabel("tries")
        plt.ylabel("seconds")
        plt.title("Methods comparison, L = " + str(L))
        plt.savefig("numpy_vs_jax_L_" + str(L) + ".png", dpi=200)
        plt.show()

        mean = np.mean(jax_time[i * tries:(i + 1) * tries])
        var = np.var(jax_time[i * tries:(i + 1) * tries])
        means.append(mean)
        std_dev.append(np.sqrt(var))

    print("JSvd time, L = " + str(L_list), file=file)
    print("jax means: " + str(means), file=file)
    print("jax std_dev: " + str(std_dev), file=file)

'''import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

command = 'export MKL_NUM_THREADS=3\nexport NUMEXPR_NUM_THREADS=3\nexport OMP_NUM_THREADS=3'
os.system(command)
'''

from auxiliary import auxiliary as aux
from auxiliary import bipartitions as bip
from states import *

from time import time
import matplotlib.pyplot as plt
from jax.numpy.linalg import eigvalsh as jneigh
from jax.scipy.linalg import eigh as jseigh


Y = 13
numb = [21, 33, 66, 129, 357]
jturbo_time = []
jscipy_time = []
jnumpy_time = []
L_list = []
tries = 10

with open("entropy_jaxes_test.txt", "a+") as file:
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
            W = matrix_from_sparse_modular_state(state, chosen, notchosen, False)

            start = time()

            l = jneigh(W.dot(W.T))
            jnumpy_time.append(time() - start)

            l = np.array(l)
            l[l < 1e-15] = 1
            entropy = np.sum(l * np.log2(l))
            print(entropy, file=file)

            start = time()

            sl = jseigh(W.dot(W.T), eigvals_only=True, turbo=False, check_finite=False, overwrite_a=True,
                        overwrite_b=True)
            jscipy_time.append(time() - start)
            sl = np.array(sl)
            sl[sl < 1e-15] = 1
            entropy = np.sum(sl * np.log2(sl))
            print(entropy, file=file)

            W = matrix_from_sparse_modular_state(state, chosen, notchosen, False)
            start = time()

            stl = jseigh(W.dot(W.T), eigvals_only=True, turbo=True, check_finite=False, overwrite_a=True,
                         overwrite_b=True)
            jturbo_time.append(time() - start)
            stl = np.array(stl)
            stl[stl < 1e-15] = 1
            entropy = np.sum(stl * np.log2(stl))
            print(entropy, file=file)

numpy_means = []
scipy_means = []
turbo_means = []
numpy_std = []
scipy_std = []
turbo_std = []

std_dev = []
L_list = []

with open("means_jax_numpy_scipy.txt", "a+") as file:
    for i in range(len(numb)):
        L = aux.lfy(numb[i])
        L_list.append(L)
        x = range(1, tries + 1)
        plt.scatter(range(1, tries + 1), jnumpy_time[i * tries:(i + 1) * tries], c='blue', marker='o',
                    label="jax numpy eigvalsh")
        plt.scatter(range(1, tries + 1), jscipy_time[i * tries:(i + 1) * tries], c='green', marker='o',
                    label="jax scipy eigh, turbo False")
        plt.scatter(range(1, tries + 1), jturbo_time[i * tries:(i + 1) * tries], c='red', marker='o',
                    label="jax scipy eigh, turbo True")

        plt.legend()
        plt.xticks(range(0, tries + 10, 10))
        plt.xlabel("tries")
        plt.ylabel("seconds")
        plt.title("Methods comparison, L = " + str(L))
        plt.savefig("jax_numpy_vs_scipy_L_" + str(L) + "_" + str(tries) + ".png", dpi=200)
        plt.show()

        numpy_mean = np.mean(jnumpy_time[i * tries:(i + 1) * tries])
        numpy_means.append(numpy_mean)
        numpy_var = np.var(jnumpy_time[i * tries:(i + 1) * tries])
        numpy_std.append(np.sqrt(numpy_var))

        scipy_mean = np.mean(jscipy_time[i * tries:(i + 1) * tries])
        scipy_means.append(scipy_mean)
        scipy_var = np.var(jscipy_time[i * tries:(i + 1) * tries])
        scipy_std.append(np.sqrt(scipy_var))

        turbo_mean = np.mean(jturbo_time[i * tries:(i + 1) * tries])
        turbo_means.append(turbo_mean)
        turbo_var = np.var(jturbo_time[i * tries:(i + 1) * tries])
        turbo_std.append(np.sqrt(turbo_var))

    print("tries: " + str(tries), file=file)
    print("numpy means: " + str(numpy_means), file=file)
    print("scipy means: " + str(scipy_means), file=file)
    print("turbo means: " + str(turbo_means), file=file)

    print("numpy std_dev: " + str(numpy_std), file=file)
    print("scipy std_dev: " + str(scipy_std), file=file)
    print("turbo std_dev: " + str(turbo_std), file=file)

from src.auxiliary.bipartitions import random_bipartition, notchosen
from src.states import *

from time import time
import matplotlib.pyplot as plt

Y = 13
numb = [21, 33, 66, 129]
np_eigval_time = []
jax_sv_time = []
L_list = []
np_sv_time = []
tries = 100

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
        l = np.linalg.eigvalsh(W.dot(W.T))
        np_eigval_time.append(time() - start)
        start = time()
        sv = numpysvd(W, compute_uv=False)
        sv2 = sv ** 2
        np_sv_time.append(time() - start)

for i in range(len(numb)):
    L = aux.lfy(numb[i])
    plt.scatter(range(1, tries + 1), np_sv_time[i * tries:(i + 1) * tries], c='red', marker='o',
                label="numpy svd")
    plt.scatter(range(1, tries + 1), np_eigval_time[i * tries:(i + 1) * tries], c='blue', marker='o',
                label="numpy eigval")

    plt.legend()
    plt.xticks(range(0, tries, 10))
    plt.xlabel("tries")
    plt.ylabel("sec")
    plt.title("Methods comparison, L = " + str(L))
    #plt.savefig("svd_vs_eigen_stability_L_" + str(L) + ".png", dpi=200)
    plt.show()

    sv_mean = np.mean(np_sv_time[i * tries:(i + 1) * tries])
    sv_var = np.var(np_sv_time[i * tries:(i + 1) * tries])
    eigval_mean = np.mean(np_eigval_time[i * tries:(i + 1) * tries])
    eigval_var = np.var(np_eigval_time[i * tries:(i + 1) * tries])

    print("Svd time, L = " + str(L))
    print(str(sv_mean) + " ± " + str(np.sqrt(sv_var)))
    print("Eigval time:")
    print(str(eigval_mean) + " ± " + str(np.sqrt(eigval_var)))
    print("")

'''
Eigval + sort
Svd time, L = 5
0.005513639450073242 ± 0.0005152399007231244
Eigval time:
0.002817845344543457 ± 0.0015741710243476835

Svd time, L = 6
0.12823628425598144 ± 0.011132972778282884
Eigval time:
0.12865104913711548 ± 0.06597776984054421

Svd time, L = 7
2.20906457901001 ± 0.3030635547047696
Eigval time:
1.2304845142364502 ± 0.6147902056516473

Svd time, L = 8
21.432851047515868 ± 0.6818328365473627
Eigval time:
47.941647295951846 ± 21.17927780740292

Eigvalh
Svd time, L = 5
0.004675657749176026 ± 0.0015160030327096867
Eigval time:
0.0014351582527160644 ± 0.0006269446465016407

Svd time, L = 6
0.10818479537963867 ± 0.016421166293532772
Eigval time:
0.05122738122940063 ± 0.006136904060813737

Svd time, L = 7
1.9086459517478942 ± 0.07740486615565235
Eigval time:
0.4030591058731079 ± 0.06359389023832983

Svd time, L = 8
20.680188720226287 ± 0.3792588573601033
Eigval time:
14.580426726341248 ± 0.26903801198702354
'''
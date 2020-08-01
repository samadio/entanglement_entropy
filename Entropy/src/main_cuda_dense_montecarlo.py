from states import *

numbers = [131] #, [15, 21,33,66,131]
maxiter = [1_600] #[450_000, 600_000, 45_000, 13_000, 1_600]
L_list = [aux.lfy(N) for N in numbers]
Y = 13
current_state = 0

files = [open("/home/samadio/entropy/Entropy/1_percent/results_L_" + str(L) + "_1.py", 'a+') for L in L_list]

for i,N in enumerate(numbers):
    print("from numpy import array", file=files[i])
    res = entanglement_entropy_montecarlo(Y, N, maxiter=maxiter[i], step=max(maxiter[i]//100, 10), tol=1e-1)
    print("results = " + str(res), file= files[i])
    files[i].close()

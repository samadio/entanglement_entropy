from entropies import entanglement_entropy_montecarlo
from states import *

numbers = [33] #, [15, 22,33,66/71,132/131/129]
maxiter = [1_000] #[1M, 1M, 500k, 50k, 12k]
L_list = [aux.lfy(N) for N in numbers]
Y = 13

#files = [open("/home/samadio/entropy/Entropy/1_percent/results_L_" + str(L) + "_" +str(maxiter[0]) +"_N_"+str(numbers[0])+".py", 'a+') for L in L_list]

for i, N in enumerate(numbers):
    #print("from numpy import array", file=files[i])
    res = entanglement_entropy_montecarlo(Y, N, maxiter=maxiter[i], step=max(maxiter[i]//100, 100), tol=1e-1, gpu=True, sparse=False)
    #print("results = " + str(res), file= files[i])
    #files[i].close()

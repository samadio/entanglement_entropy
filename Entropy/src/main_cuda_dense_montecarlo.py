from entropies import entanglement_entropy_montecarlo
from states import *
from auxiliary.auxiliary import get_candidates


Y=13
#numbers = get_candidates(13,129,256)[24:] #[15, 22,33,66/71,132/131/129]
numbers = [21]
maxiter = [100] #[1M, 1M, 500k, 50k, 12k]
L_list = [aux.lfy(N) for N in numbers]

files = [open("/home/samadio/entanglement_entropy/Entropy/1_percent/results_Y_" + str(Y) + "_N_"+str(N)+".py", 'a+') for i,N in enumerate(numbers)]

for i, N in enumerate(numbers):
    #step = max(maxiter[0]//50, 50)
    step=10

    print("from numpy import array", file=files[i])
    print("#step: " +str(step), file=files[i])
    res = entanglement_entropy_montecarlo(Y, N, maxiter=maxiter[0], step=step, tol=1e-2, gpu=False, sparse=False)
    print("results = " + str(res), file= files[i])
    files[i].close()
    if i == 0: print("lens: ", [len(r[1]) for r in res])
    #print("convergence: ", [r[0] for r in res])
    #print(str(N) + "done")

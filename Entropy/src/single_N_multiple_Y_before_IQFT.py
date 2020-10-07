from entropies import montecarlo_simulation
from states import *
from auxiliary.auxiliary import get_candidates
from math import gcd as GCD


N = 117
L = aux.lfy(N)
Y_list = [i for i in range(2,N) if GCD(i,N)==1]
maxiter = 50_000

with open("/home/samadio/entanglement_entropy/Entropy/1_percent/all_Y_N_"+str(N)+".py", 'a+') as file:

    step=1_000

    for Y in Y_list:
        nonzeros = aux.nonzeros_decimal(2 * L,Y,N)
        bipartitions = [bip.random_bipartition(range(3 * L), (3 * L) // 2) for j in range(maxiter)]
        current_state = construct_modular_state(2*L,L,nonzeros).toarray().reshape(2**(3*L))

        flags, entropies = montecarlo_simulation(current_state, step, maxiter, bipartitions, tol=1e-2, gpu=True, sparse=False)

        if flags[0] and flags[1]:
                single_Y_mean = np.mean(entropies)
                print("(" +str(Y)+ ", " +str(single_Y_mean)+ ")", file=file)
        else: print(str(Y)+" not converged")

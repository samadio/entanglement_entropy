from entropies import montecarlo_simulation
from states import *
from auxiliary.auxiliary import nonzeros_decimal
from auxiliary.bipartitions import random_bipartition

L = 8
k = 8
N = 126
Y = 13

considered_qubits = range(k+L)
bipartition_size = (k+L) // 2

maxiter = 20
 
files = [open("/home/samadio/test_L_"+str(L)+"_N_"+str(N)+"_results.py", "a+")]

nonzeros = nonzeros_decimal(k, N, Y)
state = construct_modular_state(k, L, nonzeros).toarray()

combinations = [random_bipartition(considered_qubits, bipartition_size) for j in range(maxiter)]

res = montecarlo_simulation(state, 100, maxiter, combinations, tol = 1e-1, gpu=True)
print(res, file=files[0])
files[0].close()

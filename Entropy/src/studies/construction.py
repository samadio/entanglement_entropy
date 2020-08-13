from states import *
from time import time

N = 131
L = 8
Y = 13
k = 2 * L

start = time()
state = construct_modular_state(k, L, aux.nonzeros_decimal(2 * L, N, Y)).toarray().flatten()
print("State construction time: ", time() - start)
print(log2(state.shape[0]))

chosen = bip.random_bipartition(range(k+L), (k+L)//2)
notchosen = bip.notchosen(chosen, k+L)
start = time()
W = matrix_from_state_IQFT(state, chosen, notchosen)
print("Matrix remapping time: ", time() - start)

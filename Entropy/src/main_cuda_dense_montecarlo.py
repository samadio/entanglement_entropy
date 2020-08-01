from states import *

numbers = [15] #, [21,33,66,131]
maxiter = [450_000] #[450_000, 40_000, 11_000, 900]
L_list = [aux.lfy(N) for N in numbers]
Y = 13
current_state = 0

files = [open("/home/samadio/entropy/Entropy/resullts_L=" + str(L) + ".txt", 'a+') for L in L_list]

for i,N in enumerate(numbers):
    res = entanglement_entropy_montecarlo(Y, N, maxiter=maxiter[i], step=max(maxiter[i]//100, 10))
    print(res, file= files[i])
    files[i].close()

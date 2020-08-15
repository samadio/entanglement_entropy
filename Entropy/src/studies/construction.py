from states import *
from time import time
from qiskit.quantum_info import partial_trace, entropy, Statevector
from numba import njit, prange
import itertools
from numpy.linalg import eigvalsh
from numba import cuda

try: cuda.select_device(0)
except:
    pass


@njit(parallel=True)
def first_column(chosens: np.array, n_rows: int, binary_combinations: np.array) -> np.array:
    results = np.zeros(n_rows)
    powers = np.power(2.0, np.flip(chosens))
    for c in prange(n_rows):
        results[c] = np.dot(powers, binary_combinations[c])
    return results


@njit(parallel=True)
def col_gaps(notchosens: np.array, n_cols: int, binary_combinations: np.array) -> np.array:
    results = np.zeros(n_cols)
    powers = np.power(2.0, np.flip(notchosens))
    for c in prange(n_cols):
        results[c] = np.dot(powers, binary_combinations[c])
    return results


@njit()
def get_density_matrix_from_W(W: np.array) -> np.array:
    return np.dot(W, np.conj(W).T)


def density_matrix_from_state(state: np.ndarray, chosen: list, notchosen: list) -> np.ndarray:
    n_rows = 2 ** len(chosen)
    n_cols = 2 ** len(notchosen)
    ch = np.sort(chosen)
    nch = np.sort(notchosen)

    binary_combinations = np.array(list(map(list, itertools.product([0, 1], repeat=len(ch)))), dtype=float)
    first_col_idx = np.array([first_column(np.array(ch), n_rows, binary_combinations)]).astype(int)

    binary_combinations = np.array(list(map(list, itertools.product([0, 1], repeat=len(nch)))), dtype=float)
    gaps = col_gaps(np.array(nch), n_cols, binary_combinations).astype(int)

    mappa = np.repeat(first_col_idx.T, n_cols, axis=1)
    mappa = mappa + gaps
    W = state[mappa]
    return get_density_matrix_from_W(W)


max_size = 1
N = 33
Y_list = np.unique([aux.coprime(N) for i in range(max_size)])
L = aux.lfy(N)
k = 2 * L

compute_time = []
#mapping_time = []
#create_empty_time = []
nonzero_time = []

bipartitions_tested = 10  # _000

total_start = time()

print("number of Y: ", len(Y_list))
for iid, Y in enumerate(Y_list):
    print(iid)

    or_state = construct_modular_state(k, L, aux.nonzeros_decimal(k, Y, N)).toarray().flatten()

    start = time()
    or_state = apply_IQFT(L, or_state)
    print("IQFT time: ", time() - start)

    for i in range(bipartitions_tested):
        chosen = bip.random_bipartition(range(k + L), (k + L) // 2)
        notchosen = bip.notchosen(chosen, k + L)

        start = time()
        rho = density_matrix_from_state(or_state, chosen, notchosen)
        l = eigvalsh(rho) #np.asarray(eigvalsh(rho))
        l = l[l > 1e-15]
        myreduced = -np.sum(l * np.log2(l))
        compute_time.append(time() - start)
        #print(i)
        # print("myentropy: ", myreduced)

        # W = matrix_from_state_IQFT(or_state.copy(), chosen, notchosen)

        start = time()
        qreduced = entropy(partial_trace(Statevector(or_state), notchosen))
        nonzero_time.append(time() - start)
        # print("qiskit entropy:", qreduced)

        #print("difference: ", np.abs(qreduced - myreduced))
        np.testing.assert_almost_equal(qreduced, myreduced, decimal=10)
        # np.testing.assert_array_almost_equal(W, new_W)

        # start = time()
        # row = [aux.to_decimal(aux.select_components(i, chosen)) for i in nonzero_idx_binary]
        # col = [aux.to_decimal((aux.select_components(i, notchosen))) for i in nonzero_idx_binary]
        # flatrow_idx = [i * 2 ** len(notchosen) + j for i, j in zip(row, col)]

        # a = vect_select(v_nonzero_idx_binary, chosen)
        # print(a)
        # row = vect_todec(a)
        # col = vect_todec(vect_select(v_nonzero_idx_binary, notchosen))
        # flatrow_idx = vect_flat(row, col, 2 ** len(notchosen))
        # compute_time.append(time() - start)

        # start = time()
        # W = np.zeros(len(state), dtype=complex)
        # create_empty_time.append(time() - start)

        # start = time()
        # for new_idx, old_idx in zip(flatrow_idx, nonzero_idx):
        #    W[new_idx] = state[old_idx]
        # W = W.reshape((2 ** len(chosen), 2 ** len(notchosen)))
        # mapping_time.append(time() - start)

total_end = time()
print("my time: ", np.mean(compute_time))
print("my time excluding compilation: ", np.mean(compute_time[1:]))

# print("average mapping time: ", np.mean(mapping_time))
# print("create empty time: ", np.mean(create_empty_time))
print("qiskit time: ", np.mean(nonzero_time))
print("qiskit time excluding compilation: ", np.mean(nonzero_time[1:]))

print("total time: ", total_end-total_start)
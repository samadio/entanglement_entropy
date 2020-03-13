from math import log2 as log2
import numpy as np

from auxiliary import auxiliary as aux
from scipy.sparse.coo import coo_matrix as coo_matrix
from scipy.sparse.linalg import svds as sparsesvd
from scipy.special import comb as bin_coeff
from random import sample as sample
from numpy.linalg import svd as numpysvd


def random_bipartition(sample_space: list, bipartition_size: int) -> list:
    return sample(sample_space, bipartition_size)


def notchosen(chosen: list, system_size: int) -> list:
    """
            Return array containing the qubit NOT in the partition
    :param chosen:  List of the qubits selected as partition, in the form [1,3,7,..]
    :param system_size:  Total number of qubits
    :return: notchosen:  list of qubits not chosen
    """

    if max(chosen) > system_size:
        raise ValueError("a nonexistent qubit has been chosen")

    notselected = list(set(list(range(system_size))) - set(chosen))
    return notselected


def number_of_bipartitions(size: int) -> int:
    return bin_coeff(size, size / 2, exact=True)


def create_w_from_binary(chosen: list, not_chosen: list, nonzero_binary: list, sparse: bool = True):
    """
        Return W s.t. W dot W.T is reduced density matrix according to selected bipartition

    :param chosen:      list of chosen qubits
    :param notchosen:   list of qubits to trace away
    :param nonzero_binary:  list s.t. nonzero_binary[j] = j-th index in which the state is nonzero, represented as
                            binary string:   psi = [0,1,0,1,0] -> nonzero_binary = ['001','011']
    :return: W
    """

    # row idxs of nonzero elements in W
    rows = [aux.to_decimal(aux.select_components(i, chosen)) for i in nonzero_binary]
    cols = [aux.to_decimal((aux.select_components(i, not_chosen))) for i in nonzero_binary]

    number_of_nonzeros = len(nonzero_binary)
    norm = number_of_nonzeros ** (- 1 / 2)

    data = np.ones(number_of_nonzeros) * norm
    if sparse: return coo_matrix((data, (rows, cols)), shape=(2 ** len(chosen), 2 ** len(not_chosen))).tocsc()
    flatrow_idx = [i * 2 ** len(notchosen) + j for i, j in zip(rows, cols)]
    W = np.zeros(2 ** (len(chosen) + len(notchosen)))
    W[flatrow_idx] = norm
    return W.reshape((2 ** len(chosen), 2 ** len(notchosen)))


def entropy(k: int, L: int, chosen: list, nonzero_binary: list, sparse: bool = True) -> float:
    """fixed k and bipartition"""

    not_chosen = notchosen(chosen, k + L)

    W = create_w_from_binary(chosen, not_chosen, nonzero_binary, sparse)

    if sparse:
        svds = sparsesvd(W, k=min(np.shape(W)) - 1, which='LM', return_singular_vectors=False)
    else:
        svds = numpysvd(W, compute_uv=False)
    eigs = svds * svds
    return - np.sum([i * np.log2(i) for i in eigs if i > 1e-15])


def montecarlo_single_k(k: int, L: int, nonzero_binary: list, step: int, maxiter: int = 10000) -> list:
    """
        Description
    :param k: computational step
    :param L: number of qubits in target register
    :param nonzero_binary: list containing indexes of nonzero elements, each as binary strings
    :param step: step of Montecarlo method
    :param maxiter: maximum number of iteration for Montecarlo method
    :return: results as list of entropies
    """

    qubits = range(k + L)
    partition_dimension = len(qubits) // 2
    entropies = []
    previous_mean = 0

    for i in range(maxiter):
        if (i % step == 0):
            bipartition_batch = [random_bipartition(qubits, partition_dimension) for j in range(step)]
        current_bipartition = bipartition_batch[i % step]
        current_entropy = entropy(k, L, current_bipartition, nonzero_binary)
        entropies.append(current_entropy)

        i += 1
        if i % step == 0:
            current_mean = np.mean(entropies)
            if i == step:
                previous_mean = current_mean
                continue
            tol = i ** (- 1 / 2)
            if np.abs(previous_mean - current_mean) < tol:
                return entropies
            previous_mean = current_mean

    return entropies

# def entanglement_entropy_forall_k(Y, N, step=200, sparse=True, eigen=False):
#     if (sparse == True and eigen == True): print("sparse eigen")
#     if (sparse == True and eigen == False): print("sparse svd")
#     if (sparse == False and eigen == True): print("numpy eigen")
#     if (sparse == False and eigen == False): print("numpy svd")
#
#     L = aux.lfy(N)
#     print("number of qubits: " + str(L) + "+" + str(2 * L))
#     nonzeros_decimal = aux.nonzeros_decimal(2 * L, N, Y)
#     [m * 2 ** L + (Y ** m % N) for m in range(2 ** (2 * L))]
#     print("nonzeros done")
#     results = []
#     for k in range(1, 2 * L + 1):
#         nonzero_binary = [decimal_to_binary(i, k + L) for i in nonzeros_decimal[:2 ** k]]
#         considered_qubits = range(k + L)
#         if (number_of_bipartitions(k + L) <= step):
#             results.append((k, [entropy(k, L, chosen, nonzero_binary, sparse=sparse, eigen=eigen) \
#                                 for chosen in combinations(considered_qubits, len(considered_qubits) // 2)]))
#         else:
#             results.append((k, montecarlo_single_k(k, Y, L, nonzero_binary, step, sparse=sparse, eigen=eigen)))
#             # print(str(k)+"-th computational step done")
#
#     return results

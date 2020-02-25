import math

import numpy as np
from src.auxiliary import auxiliary as aux
from scipy.sparse.coo import coo_matrix as coo_matrix
from scipy.sparse.linalg import svds as sparsesvd
from scipy.special import comb as bin_coeff
from random import random


def notchosen(chosen: list, system_size: int) -> list:
    """
            Return array containing the qubit NOT in the partition

            Parameters
            ----------
            chosen:   List
                List of the qubits selected as partition, in the form [1,3,7,..]

            system_size: Integer
                Total number of qubits

    """

    notselected = list(set(list(range(system_size))) - set(chosen))
    return notselected


def number_of_bipartitions(size):
    return bin_coeff(size, size / 2, exact=True)


def create_w(bipartition: list, not_chosen: list, nonzero_binary: list):
    """
    create matrix
    """
    row = [aux.to_decimal(aux.select_components(i, bipartition)) for i in nonzero_binary]
    col = [aux.to_decimal((aux.select_components(i, not_chosen))) for i in nonzero_binary]
    k = math.log2(len(nonzero_binary))
    data = np.ones(2 ** k) * (2 ** (- k / 2))
    return coo_matrix((data, (row, col)), shape=(2 ** len(bipartition), 2 ** len(not_chosen))).tocsc()


def entropy(k, L, bipartition, nonzero_binary):
    '''fixed k and bipartition'''

    not_chosen = notchosen(bipartition, k + L)

    # global W_time
    # t0=time.time()

    W = create_w(bipartition, not_chosen, nonzero_binary)
    # W_time.append(time.time()-t0)

    # global svd_time
    # t0=time.time()

    # if (eigen == False):
    # if (sparse):
    eigs = sparsesvd(W, \
                     k=min(np.shape(W)) - 1, which='LM', return_singular_vectors=False
                     )

    # else:
    #    eigs = numpysvd(W.toarray(), \
    #                    compute_uv=False)
    # eigs = eigs * eigs

    # if (eigen == True):
    #    if (W.shape[0] >= W.shape[1]):
    #        reduced_rho = W.T.dot(W)
    #    else:
    #        reduced_rho = W.dot(W.T)
    # reduced rho assumed hermitian
    #    if (sparse):
    #        eigs = sparse_eigsh(reduced_rho, k=min(np.shape(W)) - 1, which='LM', \
    #                            return_eigenvectors=False)
    #    else:
    #        eigs = np.linalg.eigvalsh(reduced_rho.toarray())

    # svd_time.append(time.time()-t0)

    return - np.sum([i * np.log2(i) for i in eigs if i > 1e-16])


def montecarlo_single_k(k, L, nonzero_binary, step, maxiter=10000):
    """
    fixed k, montecarlo on bipartitions
    """

    qubits = range(k + L)
    partition_dimension = len(qubits) // 2
    entropies = []
    previous_mean = 0

    for i in range(maxiter):
        if (i % step == 0):
            bipartition_batch = [random.sample(range(k + L), partition_dimension) for j in range(step)]
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

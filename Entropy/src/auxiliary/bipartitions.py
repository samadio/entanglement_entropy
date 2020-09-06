import auxiliary as aux
from scipy.sparse.coo import coo_matrix as coo_matrix
from scipy.sparse.linalg import svds as sparsesvd
from scipy.special import comb as bin_coeff
from random import sample as sample
from numpy.linalg import svd as numpysvd
import numpy as np


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


def entropy_binary(k: int, L: int, chosen: list, nonzero_binary: list, sparse: bool = True) -> float:
    """fixed k and bipartition"""

    not_chosen = notchosen(chosen, k + L)

    W = create_w_from_binary(chosen, not_chosen, nonzero_binary, sparse)

    if sparse:
        svds = sparsesvd(W, k=min(np.shape(W)) - 1, which='LM', return_singular_vectors=False)
    else:
        svds = numpysvd(W, compute_uv=False)
    eigs = svds * svds
    return - np.sum([i * np.log2(i) for i in eigs if i > 1e-15])
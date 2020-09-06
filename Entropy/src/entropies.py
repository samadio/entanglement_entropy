from states import *
from itertools import combinations as combinations
from numpy.linalg import eigvalsh as eigh

try:
    from cupy.linalg import eigvalsh as gpu_eigh
    import cupy as cp
    pack = cp
except:
    pack = np
    pass


def entanglement_entropy_from_state(state, chosen: list, sparse: bool = True, gpu: bool = False) -> float:
    """
        Compute entanglement entropy of state according to chosen bipartition of qubits

    :param state:   array representing state of the system of qubits, can be scipy.sparse or numpy depending on sparse
    :param chosen:  selected qubits
    :param sparse: True if dense representation (state is np.ndarray), False if state is a scipy.sparse.coo_matrix
    :return: S
    """

    notchosen = bip.notchosen(chosen, int(log2(state.shape[0])))

    if sparse:
        W = W_from_state_sparse(state, chosen, notchosen)
        svds = bip.sparsesvd(W, \
                             k=min(np.shape(W)) - 1, which='LM', return_singular_vectors=False)
        svds = svds ** 2
        svds = svds[svds > 1e-6]
        return - np.sum(svds * np.log2(svds))

    if gpu:
        W = cp.array(W_from_state_numba(state, chosen, notchosen))
        rho = W.dot(W.conj().T)
        eig = gpu_eigh(rho)
        eig = eig[eig > 1e-5]
        a = cp.log2(eig)
        return cp.asnumpy(- cp.sum(eig * a))

    rho = density_matrix_from_state_dense(state, chosen, notchosen)    
    eig = eigh(rho)
    eig = eig[eig > 1e-15]
    a = np.log2(eig)
    return - np.sum(eig * a)


def entanglement_entropy_montecarlo(Y: int, N: int, maxiter: int, step: int = 100, tol: float = None, gpu: bool = False,
                                    sparse: bool = True) -> list:
    """
        This function will return an approximation of bipartite entanglement entropy in Shor's Algorithm for balanced
        bipartitions. The results will be given for all the computational steps k = [1, 2L + 1]. Montecarlo methods are
        used when required. For k = [1, 2L] the computational steps consists in modular exponentiation. k = 2L + 1
        consists in the application of the IQFT on the control register.
    :param Y:       coprime of N to find the order of
    :param N:       Number to be factorized
    :param maxiter: Maximum number of iterations at which Montecarlo method stops
    :param step:    step of Montecarlo method: at least 2 * steps iteration will be computed
    :param tol:     Tolerance for convergence
    :param gpu:     Whether or not a GPU with CUDA will be used, if provided. Only available for dense representation
    :param sparse:     Whether or not to use sparse representation
    :return: S:     Entanglement entropy: S[k][1] will give entropy for (k+1)-th computation steps computed
                    on different bipartitions
    """

    L = aux.lfy(N)

    nonzeros_decimal_positions = aux.nonzeros_decimal(2 * L, Y, N)
    results = []
    current_state = 0

    bipartitions = [[bip.random_bipartition(range(k + L), (k + L) // 2) for j in range(maxiter)] for k in range(2 * L)]

    ''' Modular exponentiation  '''
    for k in range(1, 2 * L + 1):

        current_state = construct_modular_state(k, L, nonzeros_decimal_positions[:2 ** k])

        if not sparse: current_state = current_state.toarray().reshape(2 ** (k + L))
        combinations_considered = bipartitions[k - 1]

        if bip.number_of_bipartitions(k + L) <= step:
            results.append([(True, True), [entanglement_entropy_from_state(current_state, chosen, False) \
                                           for chosen in combinations(range(k + L), (k + L) // 2)]])
        else:
            results.append(
                montecarlo_simulation(current_state, step, maxiter, combinations_considered, tol=tol, gpu=gpu,
                                      sparse=sparse))

    ''' IQFT '''
    if sparse: current_state = current_state.toarray().reshape(2 ** (k + L))

    current_state = apply_IQFT(L, current_state)
    if bip.number_of_bipartitions(3 * L) <= step:
        results.append(((True, True), [entanglement_entropy_from_state(current_state, chosen, sparse=sparse, gpu=gpu) \
                                       for chosen in combinations(range(k + L), (k + L) // 2)]))
    else:
        results.append(
            montecarlo_simulation(current_state, step, maxiter, bipartitions[-1], tol=tol, gpu=gpu, sparse=sparse))

    return results


def montecarlo_simulation(state: np.array, step: int, maxiter: int, combinations_considered: list, tol: float = None,
                          gpu: bool = False, sparse=False):
    """
        Description
    :param state:                       state of the system
    :param step:                        step of Montecarlo method
    :param maxiter:                     maximum number of iteration for Montecarlo method
    :param combinations_considered:     combinations considered by the Montecarlo method
    :param tol:	    			        Tolerance for convergence
    :param gpu:                         Whether or not a GPU with CUDA will be used, if provided. Only available for dense representation
    :param sparse:                      Whether or not to use sparse representation
    :return:                            results as list of entropies
    """

    previous_mean = None
    previous_var = None
    mean_convergence = False
    var_convergence = False
    results = []
    global pack    

    if not gpu: pack = np

    for i in range(maxiter):
        current_bipartition = combinations_considered[i]
        results.append(entanglement_entropy_from_state(state, current_bipartition, sparse=sparse, gpu=gpu))

        # first step
        if i + 1 == step:
            previous_mean = pack.mean(pack.array(results))
            previous_var = pack.var(pack.array(results), ddof=1)
            continue

        if i + 1 % step == 0:
            current_mean = pack.mean(pack.array(results))
            current_var = pack.var(pack.array(results), ddof=1)

            if tol is None:
                tol = (i + 1) ** (- 1 / 2)

            mean_convergence = pack.abs(previous_mean - current_mean) < tol
            var_convergence = pack.abs(previous_var - current_var) < tol
            if mean_convergence and var_convergence: return (True, True), results
            previous_mean = current_mean
            previous_var = current_var

    return (mean_convergence, var_convergence), results

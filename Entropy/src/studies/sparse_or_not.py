import states as st
from time import time
from numpy.linalg import svd as numpysvd
from matplotlib import pyplot as plt

k = 10
L = 5
Y = 13
N = 21
number_of_qubits = k + L
nonzero_decimal = st.aux.nonzeros_decimal(k, N, Y)
state = st.construct_state(k,L,nonzero_decimal)
tries = 100
sparse_time = []
numpy_time = []
for i in range(tries):
    chosen = st.bip.random_bipartition(range(number_of_qubits),number_of_qubits // 2)
    start_time = time()
    notchosen = st.bip.notchosen(chosen, number_of_qubits)
    W_sparse = st.matrix_from_state(state, chosen, notchosen, True)
    svd = st.bip.sparsesvd(W_sparse, k=min(st.np.shape(W_sparse)) - 1, which='LM', return_singular_vectors=False)
    eigs = svd * svd
    entropy = - st.np.sum([i * st.np.log2(i) for i in eigs if i > 1e-15])
    sparse_time.append(time() - start_time)

    start_time = time()
    notchosen = st.bip.notchosen(chosen, number_of_qubits)
    W_dense = st.matrix_from_state(state, chosen, notchosen, False)
    svd = numpysvd(W_dense, compute_uv=False)
    eigs = svd * svd
    entropy = - st.np.sum([i * st.np.log2(i) for i in eigs if i > 1e-15])
    numpy_time.append(time() - start_time)

fig = plt.figure()
ax = plt.subplot(111)

plt.title("k = 10, N = 21, Y = 13")
plt.ylabel("Time (s)")
plt.xlabel("partitions")

ax.plot(range(len(sparse_time)),[i for i in sparse_time], label="sparse time",linewidth=0.4) #blue
ax.plot(range(len(numpy_time)),[i for i in numpy_time],label="numpy time",linewidth=0.4) #orange
ax.legend()
plt.savefig("Numpy_vs_Scipy_times.png", dpi=300)

print("total sparse time: " + str(sum(sparse_time)))
print("total numpy time: " + str(sum(numpy_time)))

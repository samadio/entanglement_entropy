import pyviennacl as pcl
from scipy.sparse import coo_matrix as mat
import numpy as np

A = np.eye(100)
B = pcl.CompressedMatrix(A)
tag = pcl.linalg.lanczos_tag(factor=0.99, num_eig=99, method=1, krylov=1_000)
print(B.eig(tag))

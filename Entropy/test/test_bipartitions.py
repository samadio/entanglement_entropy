from unittest import TestCase

from auxiliary.bipartitions import *
import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace
from qiskit.quantum_info import entropy as qentropy
from numpy.linalg import svd as numpysvd


class Test(TestCase):
    def test_notchosen(self):
        selected = [i * 2 for i in range(50)]
        self.assertListEqual(notchosen(selected, 100), [i * 2 + 1 for i in range(50)])
        self.assertListEqual(notchosen([1, 3], 4), [0, 2])
        with self.assertRaises(ValueError):
            notchosen([1, 3, 5, 7], 4)
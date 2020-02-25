from unittest import TestCase

from auxiliary.bipartitions import *


class Test(TestCase):
    def test_notchosen(self):
        selected = [i * 2 for i in range(50)]
        self.assertListEqual(notchosen(selected, 100), [i * 2 + 1 for i in range(50)])
        self.assertListEqual(notchosen([1, 3], 4), [0, 2])
        with self.assertRaises(ValueError):
            notchosen([1, 3, 5, 7], 4)

    def test_create_w_from_binary(self):
        # psi=[0,1/2,1/2,0,0,1/2,0.1/2], chosen=[0,2] -> nonzeros=[1,2,5,7], nonzero_binary= ['001','010','101','111']
        # rows = ['01'-> 1, 0, 3, 3]          cols= [0,1,0,1]
        # W = 1 / 2 [ [0,1] , [1,0], [0,0], [1,1] ]
        state = np.array([0, 1 / 2, 0, 1 / 2, 0, 1 / 2, 0, 1 / 2])
        chosen = [0, 2]
        nonzero = [1, 3, 5, 7]
        nonzero_binary = ['001', '011', '101', '111']
        W = create_w_from_binary(chosen, notchosen(chosen, 3), nonzero_binary).toarray()
        self.assertListEqual(W.tolist(), [[0, 0], [1 / 2, 1 / 2], [0, 0], [1 / 2, 1 / 2]])
        reduced_density_matrix = [[0, 0, 0, 0], [0, 1 / 2, 0, 1 / 2], [0, 0, 0, 0], [0, 1 / 2, 0, 1 / 2]]
        self.assertListEqual(W.dot(W.T).tolist(), reduced_density_matrix)

        chosen = [1, 2]
        W = create_w_from_binary(chosen, notchosen(chosen, 3), nonzero_binary).toarray()
        self.assertListEqual(W.dot(W.T).tolist(), reduced_density_matrix)

        chosen = [0, 1]
        W = create_w_from_binary(chosen, notchosen(chosen, 3), nonzero_binary).toarray()
        l = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
        self.assertListEqual(W.dot(W.T).tolist(), [l, l, l, l])

from unittest import TestCase

from auxiliary.auxiliary import *


class AuxiliaryTests(TestCase):
    def test_coprime(self):
        N = 225
        for i in range(10):
            Y = coprime(N)
            self.assertEqual(math.gcd(Y, N), 1)

    def test_select_components(self):
        indexes = [1, 2, 5, 7]
        data = [i for i in range(10)]
        self.assertListEqual(indexes, select_components(data, indexes))
        with self.assertRaises(ValueError):
            select_components(data, [1, 3, 5, 7, 10, 11])

    def test_lfy(self):
        self.assertEqual(lfy(123), 7)
        self.assertEqual(lfy(128), 7)
        self.assertEqual(lfy(129), 8)
        self.assertEqual(lfy(312), 9)

    def test_nonzeros_decimal(self):
        N = 21
        L = lfy(N)
        Y = 13
        k = 2 * L
        mine = nonzeros_decimal(k, Y, N)

        np.testing.assert_array_equal(mine, [nonzeros_aux(m, L, Y, N) for m in range(2 ** k)])

    def test_decimal_to_binary(self):
        nonzero = [1, 3, 5, 7]
        expected = ['001', '011', '101', '111']
        self.assertListEqual([decimal_to_binary(i,3) for i in nonzero], expected)
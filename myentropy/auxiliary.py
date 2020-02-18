from random import randint

import numpy as np
import math


def decimal_to_binary(number, length):
    """
        Returns binary representation of number as string of fixed length

        Parameters
            ----------
            number:  Integer
                     decimal number to be converted in binary form

            length:  Integer
                     minimum length of returned string

            ----------
            Example:
            decimal_to_binary(2,5) will yield "00010"
            :rtype: basestring
    """

    return np.binary_repr(number, length)


def to_decimal(string_list: list):
    """
        converts a list of char representing a binary number and returns
        its decimal representation as integer

         :param string_list:  list
                              list of binary chars representing an integer number
         :return: int
                  decimal representation of string_list

    """

    string = "".join(string_list)
    return int(string, 2)


def coprime(n: int):
    """
            Find a coprime of n for n>2
         :param n: int
                 Number to find the coprime of
         :return: int
                  a coprime of n

    """
    if n < 3:
        raise ValueError("Illegal argument: coprimes exist for N>2")
    Y = randint(2, n)
    used = []
    while 1:
        a = math.gcd(Y, n)
        if a > 1:
            # this Y is not coprime
            used.append(Y)
            while (Y in used):
                Y = randint(2, n)
        else:
            return Y


def select_components(data: list, indexes: list) -> list:
    """
             :return: list
                      slicing of data list containing only input at selected indexes
    """
    if max(indexes) not in range(len(data)):
        raise ValueError(
            'the chosen {0} bit is not present in a {1} bits register'.format(str(max(indexes)), str(len(data))))
    return [data[i] for i in indexes]


def lfy(n: int) -> int:
    return int(math.ceil(math.log2(n)))
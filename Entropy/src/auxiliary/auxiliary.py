from random import randint

import numpy as np
from math import log2, gcd, ceil
from sympy import isprime as isprime
from math import gcd as GCD

def prime_power(N):
    """
            Check if the argument N is a prime power, i.e. exist p prime and i integer s.t. p**i=N
            If the argument is is a prime power p is returned, otherwise 1 is returned
            Parameters
            ----------
            N:  Integer
            Candidate prime power

    """
    
    roof=int(ceil(log2(N))+1) #integer right after log2(N)
    
    roots=[(N**(1./i)) for i in range(2,roof)] 
    
    for (i,element) in enumerate(roots):
        if (element.is_integer() and isprime(int(element))): #then element**(i+2)=N
            return int(element)
    return 1



def get_candidates(Y: int, l_bound:int, up_bound: int) -> list:

        """
            return N odd which are not prime nor prime powers between bounds 
        :param l_bound: lower bound, included
        :param up_bound: upper bound, include
        :return: list of N
        """

        if l_bound / 2 == l_bound //2:
                l_bound+=1
        candidates = [i for i in range(l_bound,up_bound+1,2) if( (not isprime(i)) and (prime_power(i)==1) and (gcd(Y,i)==1)) ]
        return candidates


def decimal_to_binary(number: int, length: int) -> str:
    """
        Returns binary representation of number as string of fixed length
        ----------
        Example:
        decimal_to_binary(2,5) will yield "00010"

    :param number: decimal number to be converted in binary form
    :param length:  minimum length of returned string
    :return: binary representation of decimal as string of determined length
    """

    return np.binary_repr(number, length)


def to_decimal(string_list: list) -> int:
    """
        converts a list of char representing a binary number and returns its decimal representation as integer
     :param    string_list: list of binary chars representing an integer number
     :return:  decimal representation of string_list
    """

    string = "".join(string_list)
    return int(string, 2)


def coprime(n: int) -> int:
    """
            Return a coprime of n for n>2
     :param    n: Number to find the coprime of
     :return:   Y: a coprime of n
    """
    if n < 3:
        raise ValueError("Illegal argument: coprimes exist for N>2")
    Y = randint(2, n)
    used = []
    while 1:
        a = gcd(Y, n)
        if a > 1:
            # this Y is not coprime
            used.append(Y)
            while Y in used:
                Y = randint(2, n)
        else:
            return Y


def select_components(data: list, indexes: list) -> list:
    """
         Return only selected components of input
     :param     data: data to select from. Slicing must be supported
     :param     indexes: list of integers representing indexes to be selected
     :return:            slicing of data containing only input at selected indexes
    """

    if max(indexes) not in range(len(data)):
        raise ValueError(
            'the chosen {0} bit is not present in a {1} bits register'.format(str(max(indexes)), str(len(data))))
    return [data[i] for i in indexes]


def lfy(n: int) -> int:
    """
        Return number of qubits needed to represent an integer number
    :param n:  number to be represented
    :return number of qubits needed to represent n
    """
    return int(ceil(log2(n)))



def nonzeros_decimal(k: int, Y: int, N: int) -> list:
    """
        Return nonzero indexes of modular exponentiation for Y coprime of N at k-th computational step
    :param k: computational step
    :param N: number to be factorized
    :param Y: coprime of N
    :return: list of indexes of nonzero elements of state
    """

    maxlength = 2 ** k

    def modular_power_gen(max_length: int, Y: int, N: int):
        """
            generator iteratively construct nonzero indexes
        :param max_length: maximum power to generate
        :param Y: coprime
        :param N: number to factorize
        :return: m * 2 ** L + (Y** m mod N) for m in range(max_length)
        """

        L = lfy(N)
        yield 1 #Y ** 0 % N
        factor = Y % N
        # m = 1
        yield 2 ** L + factor

        res = factor
        for m in range(2, max_length):
            res = (res * factor) % N
            yield m * (2 ** L) + res

    return [i for i in modular_power_gen(maxlength, Y, N)]

import numpy as np
import math

def decimal_to_state(m,nqubit):
    '''
        Return binary representation of m as array of nqubit qubits

            Parameters
            ----------                
            m:   Integer
                number to be representend in binary form
                
            nqubit: Integer
                Total number of qubits used for representation
                    
    '''
    
    arr=np.binary_repr(m)
    arr=[int(i) for i in arr]
    if(len(arr)>nqubit):
        raise ValueError(str(nqubit)+" are not enough qubits to store the number "+str(m))
    if(len(arr)==nqubit): return arr
    return list(np.zeros(nqubit-len(arr),dtype=np.int16))+ arr


def to_decimal(array):
    '''
        Return decimal representation of the array storing a number in binary form

        Example: input [1,0,1,0,1] returns 21 
            Parameters
            ----------                
            array: List
                array containing binary representation of a number
                
                    
    '''

    size=len(array)
    return int(np.sum([array[i] *2**(size-i-1)  for i in range(size)]))

def find_coprime(N):     
    """
            Find a coprime of N for N>2
            Parameters
            ----------
            N:  Integer
                Number to find the coprime of

    """
    if(N<3):
        raise ValueError("Illegal argument: coprimes exist for N>2")
    Y=randint(2,N)
    used=[]
    while 1:
        a=math.gcd(Y,N)
        if (a>1):
            #this Y is not coprime
            used.append(Y)
            while(Y in used):
                Y=randint(2,N)
        else: return Y
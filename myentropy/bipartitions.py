import math
from math import log2 as log2
import numpy as np 
from myentropy import auxiliary as aux
from scipy.sparse.coo import coo_matrix as sparsemat
from scipy.sparse.linalg import svds as sparsesvd

def notchosen(chosen,system_size):
    """
            Return array containing the qubit NOT in the partition
            
            Parameters
            ----------                
            chosen:   List
                List of the qubits selected as partition, in the form [1,3,7,..]
                
            system_size: Integer
                Total number of qubits
                    
    """
    
    notchosen=list(set(list(range(system_size)))-set(chosen))
    notchosen.sort()
    return notchosen



def create_W(k,Y,N,chosen):
    '''
    Return the state of the quantum system in Shor's algorithm after the k-th computational step, for given N,Y.
    The state is returned as sparse matrix.
            
            Parameters
            ----------                
              
            k:      Integer
                Computational step of Shor's algorithm
            
            Y:      Integer
                Random number in [2,N-1] to find the order of

            N:      Integer
                Number to be factorized in Shor's algorithm

            chosen:   List
                List of the qubits selected as partition, in the form [1,3,7,..]
              
    
    '''
    
    L=int(math.ceil(log2(N)))
    if(k>2*L):
        raise ValueError(str(k)+"th computational step does not make sense in a "+str(2*L)+" qubits control register")
    
    #nonzero elements of psi in binary form
    nonzeros=[aux.decimal_to_state(m*2**L+(Y**m%N),k+L) for m in range(2**k)]
    not_chosen=notchosen(chosen,k+L)  

    indexes=[ (aux.to_decimal(np.take(i,chosen)),aux.to_decimal((np.take(i,not_chosen)))) for i in nonzeros]
    row=[elem[0] for elem in indexes]
    col=[elem[1] for elem in indexes]
    data=np.ones(2**k)*2**(-k/2)
    
    return sparsemat((data,(row,col)), shape=(2**len(chosen),2**len(not_chosen))    ).tocsc()


def entanglement_entropy(k,Y,N,chosen):

    '''
    This function calculates the bipartite entanglement entropy in Shor's algorithm for a given bipartition.
            
            Parameters
            ----------                
              
            k:      Integer
                Computational step of Shor's algorithm
            
            Y:      Integer
                Random number in [2,N-1] to find the order of

            N:      Integer
                Number to be factorized in Shor's algorithm

            chosen:   List
                List of the qubits selected as partition (not traced away), in the form [1,3,7,..]
              
    '''

    W=create_W(k,Y,N,chosen)
        
    eigs=np.array(sparsesvd(W,k=min(np.shape(W))-1,which='LM',return_singular_vectors=False))
    eigs=eigs*eigs
    entr=-np.sum([i*log2(i) for i in eigs if i>0])
    return entr

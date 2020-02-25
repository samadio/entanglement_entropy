import time
from Entropy.src.auxiliary.bipartitions import entanglement_entropy as entropy
from math import log2 as log2
from math import ceil as ceil
from math import gcd as GCD
import numpy as np

N=39
L=int(ceil(log2(N)))
max_k=2*L

start = time.time()
result=np.array([entropy(max_k,Y,N,[i for i in range(max_k)]) for Y in range(2,N) if(GCD(Y,N)==1)])
print(time.time()-start)
import pickle

with open("heatmap.txt", "rb") as fp:
    newdata = pickle.load(fp)

old_S=[i[2] for i in newdata[4]]

#print(np.max(np.abs(old_S-result)))

#result3=numpy.array([entropy2(k,Y,N,chosen) for k in range(1,2*L+1) for chosen in combinations(range(k+L),int((k+L)/2))])

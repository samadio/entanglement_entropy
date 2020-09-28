from auxiliary.auxiliary import prime_power as pp
from auxiliary.auxiliary import get_candidates as gc

b = gc(13,65,128)
print(b)

a = gc(13,127,256)
print(a)
print(len(a))

import numpy as np
from matplotlib import pyplot as plt
from results_L_8_12000_N_132_new import results_1, results_2

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.hist(results_2[8][1], bins=20, density=True,label="prec2")
ax2.hist(results_1[8][1],bins=20, density=True,label="prec1")
ax1.legend()
ax2.legend()
plt.savefig("test_L8_k9", dpi=300)
plt.show()

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.hist(results_2[7][1], bins=20, density=True,label="prec2")
ax2.hist(results_1[7][1],bins=20, density=True,label="prec1")
ax1.legend()
ax2.legend()
plt.savefig("test_L8_k8", dpi=300)
plt.show()

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.hist(results_2[13][1], bins=20, density=True,label="prec2")
ax2.hist(results_1[13][1],bins=20, density=True,label="prec1")
ax1.legend()
ax2.legend()
plt.savefig("test_L8_k14", dpi=300)
plt.show()


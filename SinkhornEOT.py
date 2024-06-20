# Required modules
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss

# Sinkhorn parameters
n = 100 # problem size
epsilon = 0.001 # regularization parameter
iters = 1000 # number of iterations

# Initializing marginal probability vectors
a = gauss(n, 50, 5)
b = gauss(n, 30, 10)

# Initialize cost matrix
x = np.arange(n, dtype=np.float64)
C = ot.dist(x.reshape((n,1)), x.reshape((n,1)))
C = C/C.max()

ot.plot.plot1D_mat(a,b,C, 'Cost matrix C with probability vectors a and b')

# Compute the kernel matrix
K = np.exp(-C/epsilon)

# Initial choice for the scaling variables
u_bar = np.ones(n)
v_bar = np.ones(n)

# Run the Sinkhorn algorithm
for i in range(iters):
    u_bar = a / np.dot(K, v_bar)
    v_bar = b / np.dot(np.transpose(K), u_bar)

# Compute the optimal transport plan
P = np.diag(u_bar) @ K @ np.diag(v_bar)
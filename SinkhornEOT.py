# Required modules
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
from tqdm import tqdm

# Sinkhorn parameters
n = 100 # probability vectors in \R^n
epsilon = 0.001 # regularization parameter
iters = 1000000 # number of iterations

# Initializing marginal probability vectors
a = gauss(n, 50, 5)
b = gauss(n, 30, 10)

# Initialize cost matrix
x = np.arange(n, dtype=np.float64) # vector in \R^n of the form [1,...,n]
# C = ot.dist(x.reshape((n,1)), x.reshape((n,1))) # Euclidean metric as a cost function
# Another option (Jaccard metric) for the cost function can be the following
C = ot.dist(x.reshape((n,1)), x.reshape((n,1)), metric='jaccard') 
C = C/C.max() # normalize the cost to prevent overflow in computation

print("max:",C.max(),"min:",C.min())
print(C)

# Compute the kernel matrix K_{ij} = e^{C_{ij}/\epsilon}
K = np.exp(-C/epsilon)

# Initialize Lagrange multipliers u,v to zero vectors
u_bar = np.ones(n) # e^{u_i} = 1 for all i
v_bar = np.ones(n) # e^{v_j} = 1 for all j

# visualization
ot.plot.plot1D_mat(a,b,C,'cost matrix C')
plt.savefig('jaccard_cost_matrix.png')

# iterations
for i in tqdm(range(iters)):
    u_bar = a / np.dot(K, v_bar)
    v_bar = b / np.dot(np.transpose(K), u_bar)
print("iteration is complete")

# Compute the transport plan with optimized multipliers u and v
P = np.matmul(np.matmul(np.diag(u_bar),K), np.diag(v_bar))

# Visualization
ot.plot.plot1D_mat(a,b,P,'optimal transport plan')
plt.savefig('jaccard_eot_result.png')

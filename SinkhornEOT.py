# Required modules
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import ot, itertools
import pandas as pd
import ot.plot
from ot.datasets import make_1D_gauss as gauss
from tqdm import tqdm

# Sinkhorn parameters
n = 100 # probability vectors in \R^n
epsilon = 0.001 # regularization parameter
iters = 100000 # number of iterations
filename="Quadratic_eps"+str(epsilon)+"n"+str(n)

def compute_cost_matrix_coulomb(n, N):
    # Initialize an N-dimensional array of size n in each dimension
    shape = (n,) * N
    C = np.zeros(shape)

    # Set to track already computed indices
    computed_indices = set()

    # Generate all index combinations for the upper triangular part
    for index in itertools.product(*[range(n) for _ in range(N)]):
        sorted_index = tuple(sorted(index))
        if sorted_index in computed_indices:
            continue

        total_cost = 0
        # Compute the sum of 1/abs(index[i] - index[j]) for all unique pairs (i, j)
        for i, j in itertools.combinations(range(N), 2):
            diff = np.abs(index[i] - index[j])
            if diff != 0:
                total_cost += 1 / diff
            else:
                total_cost += np.inf

        # Assign the computed cost to the corresponding element in the matrix
        C[index] = total_cost

        # Mark the indices as computed
        computed_indices.add(sorted_index)

    return C

# Initializing marginal probability vectors
a = gauss(n, 50, 5)
b = gauss(n, 30, 10)

# Initialize cost matrix
x = np.arange(n, dtype=np.float64) # vector in \R^n of the form [1,...,n]
# C = compute_cost_matrix_coulomb(n,2)
C = ot.dist(x.reshape((n,1)), x.reshape((n,1))) # Euclidean metric as a cost function
# Another option (Jaccard metric) for the cost function can be the following
# C = ot.dist(x.reshape((n,1)), x.reshape((n,1)), metric='jaccard') 
C = C/C.max() # normalize the cost to prevent overflow in computation

# Compute the kernel matrix K_{ij} = e^{C_{ij}/\epsilon}
P = np.exp(-C/epsilon)

# Initialize Lagrange multipliers u,v to zero vectors
u_bar = np.ones(n) # e^{u_i} = 1 for all i
v_bar = np.ones(n) # e^{v_j} = 1 for all j

# visualization
ot.plot.plot1D_mat(a,b,C,'cost matrix C')
plt.savefig('cost_matrix.png')

# iterations
for i in tqdm(range(iters)):
    u_bar = a*epsilon/n - (np.sum(v_bar)+np.sum(C,axis=1))/n
    v_bar = b*epsilon/n - (np.sum(u_bar)+np.sum(C,axis=0))/n
print("iteration is complete")

# Compute the transport plan with optimized multipliers u and v
for i in range(n):
    for j in range(n):
        P[i][j] = max(u_bar[i]+v_bar[j]-C[i][j],0)/epsilon
# P = np.matmul(np.matmul(np.diag(u_bar),K), np.diag(v_bar))

# Visualization
print(np.sum(P))
ot.plot.plot1D_mat(a,b,P,'optimal transport plan')
plt.savefig('eot_result'+filename+'.png')
df = pd.DataFrame(data=P.astype(float))
df.to_csv('Stochastic_P', sep= ' ', header=False, index=False)

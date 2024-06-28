from torch import torch
from matplotlib import pyplot as plt
from scipy.stats import norm
from tqdm import trange
import pandas as pd
import numpy as np
import itertools

MAX_ITER = 1e+11

def compute_cost_matrix_coulomb(n, N) -> torch.Tensor:
    # Initialize an N-dimensional array of size n in each dimension
    shape = (n,) * N
    C = torch.zeros(shape)
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
    return torch.from_numpy(C)

x = torch.linspace(-5,5,100)
# distribution generation
mu_1 = norm.pdf(x, loc=0, scale=0.4)
mu_2 = 0.5*norm.pdf(x, loc=-2, scale=0.2) + 0.5*norm.pdf(x, loc=1, scale=0.5)
mu_3 = norm.pdf(x, loc=2, scale=0.6)
mu_4 = norm.pdf(x, loc=3, scale=0.8)
mu_5 = norm.pdf(x, loc=2.75, scale=0.7)
# normalization
mu_1 = mu_1 / mu_1.sum()
mu_2 = mu_2 / mu_2.sum()
mu_3 = mu_3 / mu_3.sum()
mu_4  = mu_4 / mu_4.sum()
mu_5 = mu_5 / mu_5.sum()

def get_error(rho_1: torch.Tensor, rho_2:torch.Tensor, gamma:torch.Tensor) -> float:
    L1_rho_1 = torch.cdist(rho_1, torch.dot(gamma,torch.ones_like(rho_1)),p=1)
    L1_rho_2 = torch.cdist(rho_2, torch.dot(gamma,torch.ones_like(rho_2)),p=1)
    return max(L1_rho_1,L1_rho_2)

def quadratic_sinkhorn(C: torch.Tensor, rho_1: torch.Tensor, rho_2: torch.Tensor, epsilon: float, convergence_error: float = 1e-8, log=False) -> torch.Tensor:
    error = 1 # initialize iteration and error
    n = rho_1.size()[0] # problem size
    u, v = torch.ones(n) # initialize probability vectors
    for iter in trange(MAX_ITER):
        gamma_neg = -(u.expand_as(C.T).T + v.expand_as(C)-C).clamp_max(0)
        u=(epsilon*rho_1-(gamma_neg+u.expand_as(C)-C).sum(1))/n
        v=(epsilon*rho_2-(gamma_neg+v.expand_as(C)-C).sum(0))/n
        gamma = gamma_neg + (u.expand_as(C.T).T + v.expand_as(C) - C).clamp_min(0)/epsilon
        if (get_error(rho_1, rho_2, gamma) < convergence_error):
            break
    return gamma_neg + (u.expand_as(C.T).T + v.expand_as(C) - C).clamp_min(0)/epsilon


def ot_plot(C: torch.Tensor, rho_1: torch.Tensor, rho_2: torch.Tensor, gamma: torch.Tensor, epsilon:float, convergence_error: float, save=True):
    filename = "quadratic_epsilon={},error={}.png".format(epsilon,convergence_error)
    C, rho_1, rho_2, gamma = C.cpu(), rho_1.cpu(), rho_2.cpu(), gamma.cpu()
    fig, (ax_gamma, ax_rho_1, ax_rho_2) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot the heatmap of gamma
    ax_gamma.imshow(gamma, extent=(0, C.shape[1], 0, C.shape[0]), cmap='hot')
    ax_gamma.set_title('Joint Distribution (Gamma)')
    ax_gamma.set_ylabel('Row Index')
    ax_gamma.set_xlabel('Column Index')

    # Plot the distribution of rho_2 (histogram)
    ax_rho_2.hist(rho_2, bins=rho_2.shape[0], edgecolor='black')
    ax_rho_2.set_title('Second Marginal Distribution')
    ax_rho_2.set_xlabel('Value')
    ax_rho_2.set_ylabel('Frequency')
    ax_rho_2.tick_params(bottom=False)

    # Plot the distribution of rho_1 (histogram)
    ax_rho_1.hist(rho_1, bins=rho_1.shape[0], edgecolor='black', orientation='horizontal')
    ax_rho_1.set_title('First Marginal Distribution')
    ax_rho_1.set_xlabel('Frequency')
    ax_rho_1.set_ylabel('Value')
    ax_rho_1.tick_params(left=False)

    # Display epsilon and convergence error
    fig.suptitle(f'Epsilon: {epsilon:.4f}, Convergence Error: {convergence_error:.4e}')

    # Adjust layout
    plt.tight_layout()

    # Optionally save the plot
    if save:
      plt.savefig(filename)

    # Display the plot
    plt.show()

n = 100
C = compute_cost_matrix_coulomb(n, 2)
ot_plot(C, mu_2, mu_4,quadratic_sinkhorn(C, mu_2, mu_4, 7, 1e-9, True),7,1e-9,True)
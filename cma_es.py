import torch
from typing import Optional
from collections.abc import Callable
from tqdm.auto import tqdm


def cma_es(
        problem: Callable[[torch.Tensor], torch.Tensor],
        dim: int = 20,
        stop_fitness: float = 1e-10,
        max_epochs: int = 10000,
        device: Optional[torch.device] = None,
        maximize: bool = False,
    ) -> dict:
    """(sample_size/mu_w, lambda)-CMA-ES — based on Hansen's 'purecmaes.m'"""

    # --- Initialization ---
    dim = torch.tensor(dim, device=device)                                  # n
    mean = torch.randn(dim, device=device)                                  # m
    sigma = torch.tensor(0.3, device=device)                                # σ

    # --- Strategy parameter setting: Selection ---
    population_size = 4 + int(3 * torch.log(dim))                           # λ = 4 + ⌊3 * ln(n)⌋
    sample_size = population_size // 2                                      # μ = λ / 2
    w = torch.log(torch.tensor(sample_size + 0.5, device=device)) - \
        torch.log(torch.arange(1, sample_size + 1, device=device))          # w_i = ln(μ + 0.5) - ln(i) for i = 1,...,μ
    w /= torch.sum(w)                                                       # Σ_i w_i = 1
    effective_sample_size = torch.sum(w)**2 \
        / torch.sum(w**2)                                                   # μ_eff = (Σ_i w_i)^2 / Σ_i w_i^2

    # -- Step-size control
    c_s = (effective_sample_size + 2) / (dim + effective_sample_size + 5)                    # c_σ = (μ_eff + 2) / (n + μ_eff + 5)
    d_s = 1 + 2 * max(0, torch.sqrt((effective_sample_size - 1) / (dim + 1)) - 1) + c_s      # d_σ = 1 + 2 * max(0, sqrt((μ_eff - 1) / (n + 1)) - 1) + c_σ
    c_m = 1                                                                                  # c_m

    # --- Covariance matrix adaptation ---
    a_cov = 2                                                                                # α_cov = 2
    c_c = (4 + effective_sample_size / dim) / (dim + 4 + 2 * effective_sample_size / dim)    # c_c = (4 + μ_eff / n) / (n + 4 + 2 * μ_eff / n)
    c_1 = a_cov / ((dim + 1.3)**2 + effective_sample_size)                                   # c_1 = α_cov / ((n + 1.3)^2 + μ_eff)
    c_mu = min(
        1 - c_1,  # NOTE: On the original formula, this 0.25 factor is not present for some reason.
        a_cov * (0.25 + effective_sample_size + (1 / effective_sample_size) - 2 ) / ((dim + 2)**2 + a_cov*effective_sample_size/2)
    )                                                                                        # c_μ = min(1 - c_1, α_cov * ((1/4) + μ_eff + 1 / μ_eff - 2) / ((n + 2)^2 + α_cov*μ_eff/2))

    # --- Initialize dynamic strategy parameters ---
    p_c = torch.zeros(dim, device=device)           # Evolution path of covariance matrix for rank-one update
    p_s = torch.zeros(dim, device=device)           # Conjugate evolution path
    B = torch.eye(dim, device=device)               # Orthonormal matrix of C's eigenvectors
    D = torch.ones(dim, device=device)              # Diagonal matrix of C's singular values
    C = B @ torch.diag(D ** 2) @ B.T                # Covariance matrix C
    invsqrtC = B @ torch.diag(D ** -1) @ B.T        # C^(-1/2)
    eigeneval = 0
    chiN = torch.sqrt(torch.tensor(dim, device=device)) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))  #  This is the Expected value of ||N(0,I)|| == norm(randn(N,1))

    # --- Generation Loop ---
    g = 0                                           # Current generation
    counteval = 0                                   # Number of evaluated individuals
    while counteval < max_epochs:
        # Sample a new population of search points
        z = torch.randn(dim, population_size, device=device)
        y = B @ (D.unsqueeze(1) * z)
        x = mean.unsqueeze(1) + sigma * y

        # Evaluate the fitness of the samples and sort them (selection)
        fitness = problem(x.T)
        fitness_sort_index = torch.argsort(fitness, descending=maximize)
        fitness = fitness[fitness_sort_index]

        # Update the mean value (recombination)
        y_w = torch.sum(w * y[:, fitness_sort_index[:sample_size]], dim=1)
        mean = mean + c_m * sigma * y_w

        # Step-size control
        p_s = (1 - c_s) * p_s + torch.sqrt(c_s * (2 - c_s) * effective_sample_size) * invsqrtC @ y_w
        sigma *= torch.exp((c_s / d_s) * (torch.linalg.norm(p_s) / chiN - 1))

        # xold = mean.copy()

        # Covariance matrix adaptation
        h_s = torch.where(  # Heaviside step function
            torch.linalg.norm(p_s) / torch.sqrt(1 - (1 - c_s)**(2 * (g + 1)))
                <
            (1.4 + 2 / (dim + 1)) * chiN,
            torch.tensor(1.0, device=device),
            torch.tensor(0.0, device=device)
        )
        p_c = (1 - c_c) * p_c + h_s * torch.sqrt(c_c * (2 - c_c) * effective_sample_size) * y_w
        w_o = w * torch.where(
            w >=0,
            torch.ones_like(w),
            dim / torch.linalg.norm(invsqrtC @ y[:, fitness_sort_index[:sample_size]], dim=1)**2
        )

        # Adapt covariance matrix C
        # TODO: Check that
        C = (
            (1 - c_1 - c_mu) * C
            + c_1 * (
                torch.outer(p_c, p_c) + (1 - h_s) * c_c * (2 - c_c) * C
            )
            + c_mu * w_o * y[:, fitness_sort_index[:sample_size]] @ y[:, fitness_sort_index[:sample_size]].T
        )

        # d_h_s = (1 - h_s) * c_c * (2 - c_c)
        # C = (
        #     (1 + c_1 * d_h_s - c_1 - c_mu * torch.sum(w)) * C  # TODO: this torch.sum is wrong
        #     + c_1 * torch.outer(p_c, p_c)
        #     + c_mu * w_o * y[:, fitness_sort_index[:sample_size]] @ y[:, fitness_sort_index[:sample_size]].T
        # )

        # Update B and D from C (eigendecomposition)
        if counteval - eigeneval > population_size / (c_1 + c_mu) / dim / 10:
            eigeneval = counteval
            C = torch.triu(C) + torch.triu(C, 1).T  # enforce symmetry
            D2, B = torch.linalg.eigh(C)
            D = torch.sqrt(torch.maximum(D2, 1e-30))
            invsqrtC = torch.dot(B, torch.dot(torch.diag(D**-1), B.T))

        # --- Termination criteria ---
        if fitness[0] <= stop_fitness or torch.max(D) > 1e7 * torch.min(D):
            break

        g += 1
        counteval += population_size

    xmin = x[:, fitness_sort_index[0]]
    return xmin


if __name__ == "__main__":
    from objective_functions import rastrigin

    cma_es(rastrigin, dim=3)
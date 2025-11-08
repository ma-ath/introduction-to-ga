import torch
from typing import Optional, TypedDict
from collections.abc import Callable
from tqdm.auto import tqdm


class ResultType(TypedDict):
    best_solution: torch.Tensor
    best_fitness: torch.Tensor
    fitness_history: dict[str, list[float]]
    n_evaluations: int


def cmaes(
        problem: Callable[[torch.Tensor], torch.Tensor],
        dimension: int = 20,
        stop_fitness: float = 1e-10,
        max_epochs: int = 10000,
        device: Optional[torch.device] = None,
        maximize: bool = False,
    ) -> ResultType:
    """(sample_size/mu_w, lambda)-CMA-ES — based on Hansen's 'purecmaes.m'"""
    number_of_function_evaluations = 0

    # --- Initialization ---
    dim = torch.tensor(dimension, device=device)                            # n
    mean = torch.randn(dimension, device=device)                            # m
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
    p_c = torch.zeros(dimension, device=device)           # Evolution path of covariance matrix for rank-one update
    p_s = torch.zeros(dimension, device=device)           # Conjugate evolution path
    B = torch.eye(dimension, device=device)               # Orthonormal matrix of C's eigenvectors
    D = torch.ones(dimension, device=device)              # Diagonal matrix of C's singular values
    C = B @ torch.diag(D ** 2) @ B.T                # Covariance matrix C
    invsqrtC = B @ torch.diag(D ** -1) @ B.T        # C^(-1/2)
    eigeneval = 0
    chiN = torch.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))  #  This is the Expected value of ||N(0,I)|| == norm(randn(N,1))

    # --- Generation Loop ---
    counteval = 0                                   # Number of evaluated individuals
    best_fitness_history = []
    mean_fitness_history = []
    for g in tqdm(range(1, max_epochs // population_size + 1), desc="Evolving", unit="gen"):
        if counteval >= max_epochs:
            break
        # Sample a new population of search points
        z = torch.randn(dimension, population_size, device=device)
        y = B @ (D.unsqueeze(1) * z)
        x = mean.unsqueeze(1) + sigma * y

        # Evaluate the fitness of the samples and sort them (selection)
        fitness = problem(x.T)
        number_of_function_evaluations += x.shape[1]
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
            dim / torch.linalg.norm(invsqrtC @ y[:, fitness_sort_index[:sample_size]], dim=0)**2
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

        # NOTE: I gave up on implementing the paper version. From here on I'm following the purecmaes.m code
        # d_h_s = (1 - h_s) * c_c * (2 - c_c)
        # C = (
        #     (1 + c_1 * d_h_s - c_1 - c_mu * torch.sum(w)) * C  # TODO: this torch.sum is wrong. Should re-check the paper
        #     + c_1 * torch.outer(p_c, p_c)
        #     + c_mu * w_o * y[:, fitness_sort_index[:sample_size]] @ y[:, fitness_sort_index[:sample_size]].T
        # )

        # Update B and D from C (eigendecomposition)
        if counteval - eigeneval > population_size / (c_1 + c_mu) / dim / 10:
            eigeneval = counteval
            C = torch.triu(C) + torch.triu(C, 1).T  # enforce symmetry
            D2, B = torch.linalg.eigh(C)
            D = torch.sqrt(torch.maximum(D2, torch.tensor(1e-30, device=device)))
            invsqrtC = B @ torch.diag(D**-1) @ B.T

        mean_fitness_history.append(torch.mean(fitness).item())
        best_fitness_history.append(fitness[0].item())

        # --- Termination criteria ---
        if fitness[0] <= stop_fitness or torch.max(D) > torch.tensor(1e7, device=device) * torch.min(D):
            break

        counteval += population_size

    xmin = x[:, fitness_sort_index[0]]

    return {
        "best_solution": xmin,
        "best_fitness": fitness[0],
        "fitness_history": {
            "best": best_fitness_history,
            "mean": mean_fitness_history
        },
        "n_evaluations": number_of_function_evaluations
    }

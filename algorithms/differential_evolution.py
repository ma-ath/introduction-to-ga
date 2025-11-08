import torch
from typing import TypedDict
from collections.abc import Callable
from typing import Optional
from tqdm.auto import tqdm


class ResultType(TypedDict):
    best_solution: torch.Tensor
    best_fitness: torch.Tensor
    fitness_history: dict[str, list[float]]


def de(
        problem: Callable[[torch.Tensor], torch.Tensor],
        n_population: int = 20,
        F: float = 0.5,
        CR: float = 0.7,
        dimension: int = 10,
        search_boundary: tuple = (-5, 5),
        n_generations: int = 100,
        initial_solution: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        maximize: bool = False) -> ResultType:
    """Simple Differential Evolution (DE) implementation."""

    lower, upper = search_boundary

    # Randomly initialize population
    if initial_solution is None:
        population = torch.rand((n_population, dimension), device=device) * (upper - lower) + lower
    else:
        assert initial_solution.dim() == 1 and initial_solution.shape[0] == dimension
        population = initial_solution.unsqueeze(0).repeat(n_population, 1) + \
                        torch.rand((n_population, dimension), device=device) * 0.1  # small perturbation
        population = torch.clamp(population, lower, upper)
    fitness = problem(population)

    best_fitness_history = []
    mean_fitness_history = []

    # Optimization loop
    for _ in (pbar := tqdm(range(n_generations), desc="DE Progress")):
        # Pick three *distinct* indices not equal itself. We do this for all individuals in parallel.
        idxs = torch.arange(n_population, device=device).unsqueeze(0).repeat(n_population, 1)
        mask = torch.eye(n_population, dtype=torch.bool, device=device)
        idxs = idxs[~mask].view(n_population, n_population - 1)

        permutation = torch.stack([torch.randperm(n_population - 1, device=device)[:3] for _ in range(n_population)])
        r = idxs.gather(1, permutation)
        assert r.shape[1] >= 3
        candidates = population[r]  # Shape: (n_population, 3, dim)

        # Mutation
        mutants = candidates[:, 0] + F * (candidates[:, 1] - candidates[:, 2])
        mutants = torch.clamp(mutants, lower, upper)

        # Crossover (binomial)
        cross_points = torch.rand((n_population, dimension), device=device) < CR
        j_rand = torch.randint(0, dimension, (n_population,), device=device).unsqueeze(1)
        cross_points.scatter_(1, j_rand, True)
        trials = torch.where(cross_points, mutants, population)

        # Selection
        f_trials = problem(trials)
        if maximize:
            better_mask = f_trials > fitness
        else:
            better_mask = f_trials < fitness

        population = torch.where(better_mask.unsqueeze(1), trials, population)
        fitness = torch.where(better_mask, f_trials, fitness)

        # Record best and mean fitness
        best_idx = torch.argmin(fitness) if not maximize else torch.argmax(fitness)
        best_fitness_history.append(fitness[best_idx].item())
        mean_fitness_history.append(fitness.mean().item())

        # Update progress bar
        pbar.set_postfix({"best_fitness": best_fitness_history[-1], "mean_fitness": mean_fitness_history[-1]})

    return {
        "best_solution": population[best_idx],
        "best_fitness": fitness[best_idx],
        "fitness_history": {
            "best": best_fitness_history,
            "mean": mean_fitness_history
        }
    }

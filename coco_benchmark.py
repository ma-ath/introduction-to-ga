import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device", device)

# ---

from collections.abc import Callable
from typing import Optional
from tqdm.auto import tqdm


def differential_evolution(
        problem: Callable[[torch.Tensor], torch.Tensor],
        n_population: int = 20,
        F: float = 0.5,
        CR: float = 0.7,
        dim: int = 10,
        search_boundary: tuple = (-5, 5),
        n_generations: int = 100,
        initial_solution: Optional[torch.Tensor] = None,
        maximize: bool = False) -> dict:
    """Simple Differential Evolution (DE) implementation."""

    lower, upper = search_boundary

    # Randomly initialize population
    if initial_solution is None:
        population = torch.rand((n_population, dim), device=device) * (upper - lower) + lower
    else:
        assert initial_solution.dim() == 1 and initial_solution.shape[0] == dim
        population = initial_solution.unsqueeze(0).repeat(n_population, 1) + \
                        torch.rand((n_population, dim), device=device) * 0.1  # small perturbation
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
        cross_points = torch.rand((n_population, dim), device=device) < CR
        j_rand = torch.randint(0, dim, (n_population,), device=device).unsqueeze(1)
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

# ---

from cocoex import Problem
import numpy as np
from typing import Optional


def de_coco_wrapper(
        problem: Problem,
        x0: Optional[np.ndarray] = None,
        *,
        budget_multiplier: int = 10,
        **kwargs
    ) -> np.ndarray:
    """Wrapper to run DE on a COCO problem."""

    def objective(x_torch: torch.Tensor) -> torch.Tensor:
        """
        Define a PyTorch-compatible callable that delegates evaluation to COCO
        Convert tensor -> numpy -> evaluate -> convert back to tensor
        """
        x_np = x_torch.detach().cpu().numpy()
        fx = []
        for i in range(x_np.shape[0]):
            fx.append(problem(x_np[i]))
        fx = torch.tensor(fx, dtype=torch.float32, device=device)
        return fx

    dim = problem.dimension
    lower, upper = problem.lower_bounds, problem.upper_bounds
    budget = int(budget_multiplier * dim)
    n_population = kwargs.get("n_population", 20)
    F = kwargs.get("F", 0.5)
    CR = kwargs.get("CR", 0.7)

    result = differential_evolution(
        problem=objective,
        n_population=n_population,
        F=F,
        CR=CR,
        dim=dim,
        search_boundary=(float(lower[0]), float(upper[0])),
        n_generations=budget // n_population,
        initial_solution=torch.tensor(x0, dtype=torch.float32, device=device) if x0 is not None else None,
        maximize=kwargs.get("maximize", False)
    )

    # Return the best solution as a NumPy array
    return result["best_solution"].detach().cpu().numpy()

# ---

from cocoex import Observer, Suite, ExperimentRepeater
from cocoex.utilities import MiniPrint
from cocopp import main


suite_name = "bbob"
suite_instance = ""  # "year: 2009"
suite_options = ""  # "function_indices: 1 instance_indices: 1-10 dimensions: 2,20"
budget_multiplier = 100
n_population = 5

suite = Suite(suite_name, suite_instance, suite_options)
output_folder = f'differential_evolution_{int(budget_multiplier)}D_on_{suite_name}'
observer = Observer(suite_name, "result_folder: " + output_folder)
repeater = ExperimentRepeater(budget_multiplier)
minimal_print = MiniPrint()

while not repeater.done():
    for problem in suite:
        if repeater.done(problem):
            continue
        problem.observe_with(observer)
        problem(problem.dimension * [0])
        xopt = de_coco_wrapper(problem,
                            #    repeater.initial_solution_proposal(problem),
                               disp=False,
                               budget_multiplier=budget_multiplier,
                               n_population=n_population,
                               maximize=False)
        problem(xopt)
        repeater.track(problem)
        minimal_print(problem)

### post-process data
main(observer.result_folder + ' randomsearch! bfgs!')


# fitness x n de avaliacoes

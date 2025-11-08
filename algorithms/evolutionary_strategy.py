import torch
from typing import Callable
from tqdm.auto import tqdm
from typing import Callable, TypedDict


class ResultType(TypedDict):
    best_solution: torch.Tensor
    best_fitness: torch.Tensor
    fitness_history: dict[str, list[float]]
    n_evaluations: int


def es(
        problem: Callable[[torch.Tensor], torch.Tensor],
        dimension: int = 10,
        search_boundary: tuple = (-5, 5),
        n_generations: int = 200,
        n_parents: int = 10,
        n_offspring: int = 40,
        step_size: float = 0.1,
        plus_strategy: bool = True,
        maximize: bool = False) -> ResultType:
    """Simple Evolutionary Strategy (ES) implementation."""
    number_of_function_evaluations = 0

    lower, upper = search_boundary

    # Randomly initialize population
    parents = torch.rand((n_parents, dimension)) * (upper - lower) + lower
    fitness = problem(parents)
    number_of_function_evaluations += parents.shape[0]

    best_fitness_history = []
    mean_fitness_history = []

    # === Main loop ===
    for _ in (pbar := tqdm(range(n_generations), desc="ES Progress")):
        # --- Generate λ offspring ---
        parents_indices = torch.randint(0, n_parents, (n_offspring,))
        parent = parents[parents_indices]
        offspring = parent + step_size * torch.randn(n_offspring, dimension)
        offspring = torch.clamp(offspring, lower, upper)
        offspring_fitness = problem(offspring)
        number_of_function_evaluations += offspring.shape[0]

        # --- Selection ---
        if plus_strategy:
            # (μ + λ)-ES → parents compete with offspring
            combined = torch.vstack((parents, offspring))
            combined_fitness = torch.hstack((fitness, offspring_fitness))
        else:
            # (μ, λ)-ES → only offspring survive
            combined = offspring
            combined_fitness = offspring_fitness

        idx = torch.argsort(combined_fitness, descending=maximize)
        parents = combined[idx[:n_parents]]
        fitness = combined_fitness[idx[:n_parents]]

        best_fitness_history.append(torch.min(fitness))
        mean_fitness_history.append(torch.mean(fitness))

        pbar.set_postfix({
            "best_fitness": best_fitness_history[-1].item(),
            "mean_fitness": mean_fitness_history[-1].item()
        })

    # Return best solution found
    if maximize:
        best_idx = torch.argmax(fitness)
    else:
        best_idx = torch.argmin(fitness)

    return {
        "best_solution": parents[best_idx],
        "best_fitness": fitness[best_idx],
        "fitness_history": {
            "best": best_fitness_history,
            "mean": mean_fitness_history
        },
        "n_evaluations": number_of_function_evaluations
    }

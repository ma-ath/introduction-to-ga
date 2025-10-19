from typing import Callable
from tqdm.auto import tqdm
import numpy as np
import os


def evolutionary_strategy(problem: Callable,
              vector_size: int = 10,
              bounds: tuple = (-5, 5),
              generations: int = 200,
              mu: int = 10,
              lam: int = 40,
              sigma: float = 0.1,
              plus_strategy: bool = True):
    """
    Simple (μ + λ)-ES or (μ, λ)-ES with fixed step-size (no self-adaptation)

    problem: objective function to minimize
    vector_size: number of parameters
    bounds: (lower, upper)
    generations: number of iterations
    mu: number of parents
    lam: number of offspring
    sigma: fixed mutation step size
    plus_strategy: True for (μ + λ)-ES, False for (μ, λ)-ES
    """

    lower, upper = bounds

    # === Initialize population ===
    parents = np.random.uniform(lower, upper, (mu, vector_size))
    fitness = np.array([problem(p) for p in parents])

    best_fitness_history = []
    mean_fitness_history = []

    # === Main loop ===
    for _ in tqdm(range(generations), desc="ES Generations"):
        offspring = []

        # --- Generate λ offspring ---
        for _ in range(lam):
            parent = parents[np.random.randint(mu)]
            child = parent + sigma * np.random.randn(vector_size)
            child = np.clip(child, lower, upper)
            offspring.append(child)

        offspring = np.array(offspring)
        offspring_fitness = np.array([problem(o) for o in offspring])

        # --- Selection ---
        if plus_strategy:
            # (μ + λ)-ES → parents compete with offspring
            combined = np.vstack((parents, offspring))
            combined_fitness = np.hstack((fitness, offspring_fitness))
        else:
            # (μ, λ)-ES → only offspring survive
            combined = offspring
            combined_fitness = offspring_fitness

        idx = np.argsort(combined_fitness)
        parents = combined[idx[:mu]]
        fitness = combined_fitness[idx[:mu]]

        best_fitness_history.append(np.min(fitness))
        mean_fitness_history.append(np.mean(fitness))

    # === Return best solution ===
    best_idx = np.argmin(fitness)
    return parents[best_idx], fitness[best_idx], (best_fitness_history, mean_fitness_history)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
   
    # Example Problem
    def sphere(x: np.ndarray) -> float:
        return sum(x**2)

    best, best_fitness, history = evolutionary_strategy(sphere)
    best_fitness_history, mean_fitness_history = history

    print("Best Fitness:", best_fitness)
    print("Best Individual:", best)

    # Plotting fitness history
    plt.plot(best_fitness_history, label='Best Fitness')
    plt.plot(mean_fitness_history, label='Mean Fitness')
    plt.yscale('log')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.title('Evolutionary Strategy Optimization')
    out_path = os.path.join(os.path.dirname(__file__), "es_optimization.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

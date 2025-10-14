from typing import Callable
from tqdm.auto import tqdm
import numpy as np


def differential_evolution(problem: Callable,
                           NP: int = 20,
                           F: float = 0.5,
                           CR: float = 0.7,
                           vector_size: int = 10,
                           bounds: tuple = (-1, 1),
                           generations: int = 100):
    """
    problem: objective, takes a 1D array and returns scalar (to minimize)
    NP: number of individuals in population
    F: mutation scaling factor (commonly ~0.5)
    CR: crossover probability
    generations: number of generations
    """
    # Initialize population
    lower, upper = bounds
    population = []
    for _ in range(NP):
        individual = np.random.randn(vector_size)
        individual = np.clip(individual, lower, upper)
        population.append(individual)
    
    # Evaluate
    fitness = np.array([problem(individual) for individual in population])

    # DE main loop
    best_fitness_history = []
    mean_fitness_history = []
    for _ in tqdm(range(generations), desc="DE Generations"):
        for i in range(NP):
            # pick three *distinct* indices not equal i
            idxs = [idx for idx in range(NP) if idx != i]
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            x1, x2, x3 = population[r1], population[r2], population[r3]

            # Mutation
            mutant = x1 + F * (x2 - x3)
            mutant = np.clip(mutant, lower, upper)

            # Crossover (binomial)
            cross_points = np.random.rand(vector_size) < CR
            j_rand = np.random.randint(vector_size)
            cross_points[j_rand] = True
            trial = np.where(cross_points, mutant, population[i])
            
            # Selection
            f_trial = problem(trial)
            if f_trial < fitness[i]:
                population[i] = trial
                fitness[i] = f_trial

        # Record best and mean fitness
        best_idx = np.argmin(fitness)
        best_fitness_history.append(fitness[best_idx])
        mean_fitness_history.append(np.mean(fitness))

    best_idx = np.argmin(fitness)
    best_individual = population[best_idx]
    best_fitness = fitness[best_idx]
    return best_individual, best_fitness, (best_fitness_history, mean_fitness_history)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
   
    # Example Problem
    def sphere(x: np.ndarray) -> float:
        return sum(x**2)

    best, best_fitness, history = differential_evolution(sphere)
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
    plt.title('Differential Evolution Optimization')
    plt.show()

import matplotlib.pyplot as plt

from evolutionary_strategy import evolutionary_strategy
from problems import sphere, rosenbrock, rastrigin, ackley, griewank, schwefel, levy


def main():
    problems = {
        "Sphere": sphere,
        "Rosenbrock": rosenbrock,
        "Rastrigin": rastrigin,
        "Ackley": ackley,
        "Griewank": griewank,
        "Schwefel": schwefel,
        "Levy": levy,
    }

    for name, problem in problems.items():
        print(f"Optimizing {name} function...")
        best, best_fitness, history = evolutionary_strategy(problem, bounds=(-500, 500), vector_size=10, mu=50, generations=10000, lam=100, sigma=1, plus_strategy=True)
        best_fitness_history, mean_fitness_history = history
        print(f"Best Fitness: {best_fitness}")
        print(f"Best Individual: {best}\n")

        plt.figure(figsize=(10, 5))
        plt.plot(best_fitness_history, label='Best Fitness')
        plt.plot(mean_fitness_history, label='Mean Fitness')
        plt.title(f'Fitness over Generations for {name} Function')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.grid()
        plt.savefig(f"es_optimization_{name}.png", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main()
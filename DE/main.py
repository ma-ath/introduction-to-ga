import matplotlib.pyplot as plt

from differential_evolution import differential_evolution
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
        best, best_fitness, history = differential_evolution(problem, bounds=(-100, 100), NP=50, generations=1000)
        best_fitness_history, mean_fitness_history = history
        print(f"Best Fitness: {best_fitness}")
        print(f"Best Individual: {best}\n")

        plt.figure(figsize=(10, 5))
        plt.plot(best_fitness_history, label='Best Fitness')
        plt.plot(mean_fitness_history, label='Mean Fitness')
        plt.title(f'Fitness over Generations for {name} Function')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        # plt.yscale('log')
        plt.legend()
        plt.grid()
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()
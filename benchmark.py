if __name__ == "__main__":
    import torch
    from logger import setup_logger
    from algorithms import es, de, cmaes
    from objective_functions import sphere, rosenbrock, rastrigin, ackley, griewank, schwefel, levy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(console_level="debug", file_level=None, telegram_level=None)
    logger.debug(f"Using device: {device}")

    # Define algorithms and their parameters
    algorithms = {
        "es": {
            "name": "Evolutionary Strategy",
            "implementation": es,
            "kwargs": {
                "dimension": 3,
                "max_problem_evaluations": 200,
                "stop_fitness": 1e-10,
                "n_parents": 10,
                "n_offspring": 40,
                "step_size": 0.1,
                "plus_strategy": True,
                "maximize": False
            }
        },
        "de": {
            "name": "Differential Evolution",
            "implementation": de,
            "kwargs": {
                "dimension": 3,
                "max_problem_evaluations": 200,
                "stop_fitness": 1e-10,
                "n_population": 23,
                "F": 0.5,
                "CR": 0.7,
                "initial_solution": None,
                "maximize": False
            }
        },
        "cma-es": {
            "name": "Covariance Matrix Adaptation - Evolution Strategy",
            "implementation": cmaes,
            "kwargs": {
                "dimension": 3,
                "max_problem_evaluations": 200,
                "stop_fitness": 1e-10,
                "max_epochs": 10000,
                "device": device,
                "maximize": False,
            }
        }
    }

    # Define objective functions to test
    objective_functions = {
        "Sphere": {
            "function": sphere,
            "search_boundary": (-5, 5),
        },
        "Rastrigin": {
            "function": rastrigin,
            "search_boundary": (-5.12, 5.12)
        },
        "Rosenbrock": {
            "function": rosenbrock,
            "search_boundary": (-2, 2)
        },
        "Ackley": {
            "function": ackley,
            "search_boundary": (-10, 10)
        },
        "Schwefel": {
            "function": schwefel,
            "search_boundary": (-500, 500)
        },
        "Griewank": {
            "function": griewank,
            "search_boundary": (-100, 100)
        },
        "Levy": {
            "function": levy,
            "search_boundary": (-10, 10)
        }
    }

    # Dictionary to store results
    results = {}

    # Run benchmarks
    for algorithm_name, algorithm_info in algorithms.items():
        results[algorithm_name] = {}
        for objective_function_name, objective_function_info in objective_functions.items():
            logger.info(f"Testing algorithm: \"{algorithm_info['name']}\" on function: \"{objective_function_name}\"")

            solution = algorithm_info["implementation"](
                problem=objective_function_info["function"],
                search_boundary=objective_function_info["search_boundary"],
                **algorithm_info["kwargs"]
            )
            results[algorithm_name][objective_function_name] = solution
            logger.info(f"Best fitness {solution['best_fitness']} found with {solution['n_evaluations']} evaluations.")

if __name__ == "__main__":
    import torch
    from logger import setup_logger
    from algorithms import es, de, cmaes
    from objective_functions import sphere, rosenbrock, rastrigin, ackley, griewank, schwefel, levy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(__name__, console_level=None, file_level="info", telegram_level=None)
    logger.debug(f"Using device: {device}")


    algorithms = {
        "es": {
            "name": "Evolutionary Strategy",
            "implementation": es,
            "kwargs": {
                "dimension": 20,
                "search_boundary": (-5, 5),
                "n_generations": 200,
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
                "dimension": 20,
                "n_population": 20,
                "F": 0.5,
                "CR": 0.7,
                "search_boundary": (-5, 5),
                "n_generations": 100,
                "initial_solution": None,
                "maximize": False
            }
        },
        "cma-es": {
            "name": "Covariance Matrix Adaptation - Evolution Strategy",
            "implementation": cmaes,
            "kwargs": {
                "dimension": 20,
                "stop_fitness": 1e-10,
                "max_epochs": 10000,
                "device": device,
                "maximize": False,
            }
        }
    }
    objective_functions = [sphere, rastrigin, rosenbrock, ackley, schwefel, griewank, levy]
    results = {}
    for algorithm_name, algorithm_info in algorithms.items():
        results[algorithm_name] = {}
        for objective_function in objective_functions:
            logger.info(f"Testing algorithm: {algorithm_name} on function: {objective_function.__name__}")

            solution = algorithm_info["implementation"](
                problem=objective_function,
                **algorithm_info["kwargs"]
            )
            results[algorithm_name][objective_function.__name__] = solution

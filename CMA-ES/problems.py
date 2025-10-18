import numpy as np


def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def rastrigin(x):
    return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))

def ackley(x):
    n = len(x)
    return -20*np.exp(-0.2*np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2*np.pi*x))) + 20 + np.e

def griewank(x):
    return np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1)))) + 1

def schwefel(x):
    return 418.9829*len(x) - np.sum(x*np.sin(np.sqrt(np.abs(x))))

def levy(x):
    w = 1 + (x - 1)/4
    term1 = np.sin(np.pi*w[0])**2
    term3 = (w[-1]-1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
    term2 = np.sum((w[:-1]-1)**2 * (1 + 10*np.sin(np.pi*w[:-1]+1)**2))
    return term1 + term2 + term3


if __name__ == "__main__":
    # Example usage
    dim = 10
    bounds = [(-5, 5)] * dim
    print("Sphere:", sphere(np.random.uniform(-5, 5, dim)))
    print("Rosenbrock:", rosenbrock(np.random.uniform(-5, 5, dim)))
    print("Rastrigin:", rastrigin(np.random.uniform(-5, 5, dim)))
    print("Ackley:", ackley(np.random.uniform(-5, 5, dim)))
    print("Griewank:", griewank(np.random.uniform(-600, 600, dim)))
    print("Schwefel:", schwefel(np.random.uniform(-500, 500, dim)))
    print("Levy:", levy(np.random.uniform(-10, 10, dim)))

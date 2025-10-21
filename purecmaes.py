import numpy as np


def frosenbrock(x):
    """Rosenbrock function (min at x = [1, 1, ..., 1])"""
    x = np.asarray(x)
    return np.sum(100.0 * (x[:-1]**2 - x[1:])**2 + (x[:-1] - 1.0)**2)


def pure_cmaes():
    """(mu/mu_w, lambda)-CMA-ES — direct translation of Hansen's 'purecmaes.m'"""

    # --- Initialization ---
    strfitnessfct = frosenbrock      # objective function
    N = 20                           # dimension
    xmean = np.random.rand(N)        # initial mean
    sigma = 0.3                      # initial step size
    stopfitness = 1e-10              # stop if fitness < stopfitness
    stopeval = 1e3 * N**2            # maximum number of evaluations

    # --- Strategy parameter setting: Selection ---
    lam = 4 + int(3 * np.log(N))     # population size
    mu = lam // 2                    # number of parents
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)       # normalize
    mueff = np.sum(weights)**2 / np.sum(weights**2)

    # --- Adaptation parameters ---
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3)**2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2)**2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs

    # --- Initialize dynamic strategy parameters ---
    pc = np.zeros(N)
    ps = np.zeros(N)
    B = np.eye(N)
    D = np.ones(N)
    C = np.dot(B, np.dot(np.diag(D**2), B.T))
    invsqrtC = np.dot(B, np.dot(np.diag(D**-1), B.T))
    eigeneval = 0
    chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N**2))

    # --- Generation Loop ---
    counteval = 0
    while counteval < stopeval:
        # Generate and evaluate lambda offspring
        arz = np.random.randn(N, lam)
        arx = xmean[:, None] + sigma * np.dot(B, (D[:, None] * arz))
        arfitness = np.apply_along_axis(strfitnessfct, 0, arx)
        counteval += lam

        # Sort by fitness and compute new mean
        arindex = np.argsort(arfitness)
        arfitness = arfitness[arindex]
        xold = xmean.copy()
        xmean = np.dot(arx[:, arindex[:mu]], weights)

        # Cumulation: Update evolution paths
        y = (xmean - xold) / sigma
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.dot(invsqrtC, y)
        hsig = (np.linalg.norm(ps) /
                np.sqrt(1 - (1 - cs)**(2 * counteval / lam)) / chiN
                < (1.4 + 2 / (N + 1)))
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y

        # Adapt covariance matrix C
        artmp = (1 / sigma) * (arx[:, arindex[:mu]] - xold[:, None])
        C = ((1 - c1 - cmu) * C
             + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * np.dot(artmp * weights, artmp.T))

        # Adapt step size sigma
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        # Update B and D from C (eigendecomposition)
        if counteval - eigeneval > lam / (c1 + cmu) / N / 10:
            eigeneval = counteval
            C = np.triu(C) + np.triu(C, 1).T  # enforce symmetry
            D2, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D2, 1e-30))
            invsqrtC = np.dot(B, np.dot(np.diag(D**-1), B.T))

        # --- Termination criteria ---
        if arfitness[0] <= stopfitness or np.max(D) > 1e7 * np.min(D):
            break

    xmin = arx[:, arindex[0]]
    return xmin


if __name__ == "__main__":
    best_solution = pure_cmaes()
    print("Best solution found:", best_solution)
    print("Best fitness:", frosenbrock(best_solution))

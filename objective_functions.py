import torch


def sphere(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() <= 2, "Input tensor must be 1D or 2D"
    if x.dim() == 2:
        return torch.sum(x**2, dim=1)
    return torch.sum(x**2)

def rosenbrock(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() <= 2, "Input tensor must be 1D or 2D"
    if x.dim() == 2:
        return torch.sum(100*(x[:, 1:] - x[:, :-1]**2)**2 + (x[:, :-1] - 1)**2, dim=1)
    return torch.sum(100*(x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def rastrigin(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() <= 2, "Input tensor must be 1D or 2D"
    if x.dim() == 2:
        return 10*x.size(1) + torch.sum(x**2 - 10*torch.cos(2*torch.pi*x), dim=1)
    return 10*len(x) + torch.sum(x**2 - 10*torch.cos(2*torch.pi*x))

def ackley(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() <= 2, "Input tensor must be 1D or 2D"
    if x.dim() == 2:
        n = x.size(1)
        return -20*torch.exp(-0.2*torch.sqrt(torch.mean(x**2, dim=1))) - torch.exp(torch.mean(torch.cos(2*torch.pi*x), dim=1)) + 20 + torch.e
    n = len(x)
    return -20*torch.exp(-0.2*torch.sqrt(torch.mean(x**2))) - torch.exp(torch.mean(torch.cos(2*torch.pi*x))) + 20 + torch.e

def griewank(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() <= 2, "Input tensor must be 1D or 2D"
    if x.dim() == 2:
        return torch.sum(x**2, dim=1)/4000 - torch.prod(torch.cos(x/torch.sqrt(torch.arange(1, x.size(1)+1).unsqueeze(0))), dim=1) + 1
    return torch.sum(x**2)/4000 - torch.prod(torch.cos(x/torch.sqrt(torch.arange(1, len(x)+1)))) + 1

def schwefel(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() <= 2, "Input tensor must be 1D or 2D"
    if x.dim() == 2:
        return 418.9829*x.size(1) - torch.sum(x*torch.sin(torch.sqrt(torch.abs(x))), dim=1)
    return 418.9829*len(x) - torch.sum(x*torch.sin(torch.sqrt(torch.abs(x))))

def levy(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() <= 2, "Input tensor must be 1D or 2D"
    w = 1 + (x - 1)/4
    if x.dim() == 2:
        term1 = torch.sin(torch.pi*w[:, 0])**2
        term3 = (w[:, -1]-1)**2 * (1 + torch.sin(2*torch.pi*w[:, -1])**2)
        term2 = torch.sum((w[:, :-1]-1)**2 * (1 + 10*torch.sin(torch.pi*w[:, :-1]+1)**2), dim=1)
        return term1 + term2 + term3
    term1 = torch.sin(torch.pi*w[0])**2
    term3 = (w[-1]-1)**2 * (1 + torch.sin(2*torch.pi*w[-1])**2)
    term2 = torch.sum((w[:-1]-1)**2 * (1 + 10*torch.sin(torch.pi*w[:-1]+1)**2))
    return term1 + term2 + term3


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import os

    # Define input tensors for testing
    dim = 3
    batch = 10
    input_tensor = torch.rand(dim)
    batch_input = input_tensor.unsqueeze(0).repeat(batch, 1)

    # Number of points for plotting
    n_plot_points = 2000

    # Create directory for plots
    if not os.path.exists("function_plots"):
        os.makedirs("function_plots")

    # Test each function and generate plots
    for function, boundaries in zip([sphere, rosenbrock, rastrigin, ackley, griewank, schwefel, levy],
                                     [(-5, 5), (-2, 2), (-5.12, 5.12), (-10, 10), (-100, 100), (-500, 500), (-10, 10)]):
        print(f"Testing function: {function.__name__}")
        print(f"{function.__name__}: {function(input_tensor)}")
        print(f"{function.__name__} (batch): {function(batch_input)}")

        # Plotting
        print("Generating plot for 3D surface...")
        
        # Create a grid for 3D plotting
        x_min, x_max = boundaries
        x = torch.linspace(x_min, x_max, n_plot_points)
        y = torch.linspace(x_min, x_max, n_plot_points)
        xx, yy = torch.meshgrid(x, y, indexing="xy")

        # Stack the grid into a 2D tensor: shape (N*N, 2)
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


        z = function(grid)  # Evaluate in batch
        z = z.reshape(n_plot_points, n_plot_points).detach().numpy()

        # Convert grid to numpy
        Xn, Yn = xx.numpy(), yy.numpy()

        # 3D plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(Xn, Yn, z, cmap="viridis", linewidth=0, antialiased=True)

        ax.set_title(f"{function.__name__.capitalize()} Function")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_zlabel("f(x₁, x₂)")

        plt.tight_layout()

        plt.savefig(f"function_plots/{function.__name__}_3d_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved as function_plots/{function.__name__}_3d_plot.png\n")

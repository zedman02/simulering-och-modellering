import numpy as np
import matplotlib.pyplot as plt

def relax(grid, grid_new, n):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            grid_new[i, j] = 0.25 * (grid[i + 1, j] + grid[i - 1, j] + grid[i, j + 1] + grid[i, j - 1])

# Function to calculate the maximum error
def calculate_error(grid, grid_new, n):
    return np.max(np.abs(grid_new[1:-1, 1:-1] - grid[1:-1, 1:-1]))

# Function to initialize the grid with custom boundary conditions
def initialize_grid_with_custom_boundaries(n):
    """
    Initialize the grid with specific boundary conditions:
    - Top and bottom sides: 10
    - Left and right sides: 5
    """
    # Initialize the potential grid
    v = np.zeros((n + 2, n + 2))  # Include boundary points
    vnew = np.zeros_like(v)

    # Set boundary conditions
    v[0, :] = 10  # Bottom
    v[-1, :] = 0  # Top boundary
    v[:, 0] = 10  # Left boundary
    v[:, -1] = 10  # Right boundary

    # Initialize interior points to a reasonable guess (average of boundary values)
    v[1:-1, 1:-1] = 7.5

    return v, vnew

# Function to relax the grid and compute the solution
def relax_until_converged(v, vnew, n, tolerance=0.01):
    iterations = 0
    max_error = tolerance + 1  # Initialize with a value greater than tolerance

    while max_error > tolerance:
        relax(v, vnew, n)
        max_error = calculate_error(v, vnew, n)
        v[1:-1, 1:-1] = vnew[1:-1, 1:-1]  # Update interior points
        iterations += 1

    return v, iterations

# Function to plot equipotential surfaces
def plot_equipotential(grid, title="Equipotential Surfaces"):
    plt.figure(figsize=(8, 6))
    plt.contourf(grid, levels=20, cmap='viridis')
    plt.colorbar(label="Potential (V)")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Analyze for 9x9 grid
n_9x9 = 9
v_9x9, vnew_9x9 = initialize_grid_with_custom_boundaries(n_9x9)
final_grid_9x9, iterations_9x9 = relax_until_converged(v_9x9, vnew_9x9, n_9x9)

# Plot results for 9x9 grid
plot_equipotential(final_grid_9x9, title="Equipotential Surfaces (9x9 Grid)")

# Analyze for 20x20 grid
n_20x20 = 20
v_20x20, vnew_20x20 = initialize_grid_with_custom_boundaries(n_20x20)
final_grid_20x20, iterations_20x20 = relax_until_converged(v_20x20, vnew_20x20, n_20x20)

# Plot results for 20x20 grid
plot_equipotential(final_grid_20x20, title="Equipotential Surfaces (20x20 Grid)")

# Print the number of iterations
print("Iterations for 9x9 grid:", iterations_9x9)
print("Iterations for 20x20 grid:", iterations_20x20)

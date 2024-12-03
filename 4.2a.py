import numpy as np
import matplotlib.pyplot as plt

# Function to initialize the grid with custom boundary conditions
def initialize_grid(n):
    """
    Initialize the grid with the following boundary conditions:
    - Top and bottom sides: 10
    - Left and right sides: 5
    """
    v = np.zeros((n + 2, n + 2))  # Include boundary points
    

    # Set boundary conditions
    v[0, :] = 10  # Bottom
    v[-1, :] = 0  # Top boundary
    v[:, 0] = 10  # Left boundary
    v[:, -1] = 10  # Right boundary

    # Initialize interior points to a reasonable guess (average of boundary values)
    v[1:-1, 1:-1] = 7.5

    """for i in range(1, n + 1):
        v[i, 1:-1] = 10 * (1 - i / (n + 1))"""

    """center_x = (n + 1) // 2  # Center of the grid in x-direction
    center_y = (n + 1) // 2  # Center of the grid in y-direction
    v[center_x, center_y] = 4"""

    return v

# Gauss-Seidel relaxation method
def relax_gauss_seidel(v, n):
    """
    Update the potential grid using the Gauss-Seidel method.
    Updates are applied sequentially, immediately using the most recent values.
    """
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            v[i, j] = 0.25 * (v[i + 1, j] + v[i - 1, j] + v[i, j + 1] + v[i, j - 1])

# Function to compute the maximum error
def calculate_error_gauss_seidel(v, n):
    """
    Compute the maximum error based on the difference between old and new values.
    """
    max_error = 0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            updated_value = 0.25 * (v[i + 1, j] + v[i - 1, j] + v[i, j + 1] + v[i, j - 1])
            error = abs(updated_value - v[i, j])
            max_error = max(max_error, error)
    return max_error

# Perform Gauss-Seidel relaxation until convergence
def gauss_seidel_until_converged(v, n, tolerance=0.01):
    """
    Perform Gauss-Seidel relaxation until the maximum error is below the tolerance.
    """
    iterations = 0
    while True:
        max_error = calculate_error_gauss_seidel(v, n)
        if max_error < tolerance:
            break
        relax_gauss_seidel(v, n)
        iterations += 1
    return v, iterations

# Initialize grid for 9x9 and 20x20 grids
n_9x9 = 9
n_20x20 = 20

v_9x9 = initialize_grid(n_9x9)
v_20x20 = initialize_grid(n_20x20)

# Perform Gauss-Seidel relaxation for both grids
final_grid_9x9, iterations_9x9 = gauss_seidel_until_converged(v_9x9, n_9x9)
final_grid_20x20, iterations_20x20 = gauss_seidel_until_converged(v_20x20, n_20x20)

# Display the number of iterations
print(f"Iterations for 9x9 grid (Gauss-Seidel): {iterations_9x9}")
print(f"Iterations for 20x20 grid (Gauss-Seidel): {iterations_20x20}")

# Plot the results for visualization
def plot_equipotential(grid, title):
    plt.figure(figsize=(8, 6))
    plt.contourf(grid, levels=20, cmap='viridis')
    plt.colorbar(label='Potential (V)')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

plot_equipotential(final_grid_9x9, "Equipotential Surfaces (9x9 Grid - Gauss-Seidel)")
plot_equipotential(final_grid_20x20, "Equipotential Surfaces (20x20 Grid - Gauss-Seidel)")

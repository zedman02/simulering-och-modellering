import numpy as np
import matplotlib.pyplot as plt

# Initialize the grid with boundary conditions
def initialize_grid(n):
    """
    Initialize the grid with boundary conditions:
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
    
    center_x = (n + 1) // 2  # Center of the grid in x-direction
    center_y = (n + 1) // 2  # Center of the grid in y-direction
    v[center_x, center_y] = 4

    return v

# Checkerboard relaxation method
def relax_checkerboard(v, n):
    """
    Update the grid using the checkerboard method:
    - Update red sites first (i + j is even).
    - Update black sites next (i + j is odd).
    """
    # Update red sites
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if (i + j) % 2 == 0:  # Red sites
                v[i, j] = 0.25 * (v[i + 1, j] + v[i - 1, j] + v[i, j + 1] + v[i, j - 1])

    # Update black sites
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if (i + j) % 2 == 1:  # Black sites
                v[i, j] = 0.25 * (v[i + 1, j] + v[i - 1, j] + v[i, j + 1] + v[i, j - 1])

# Function to calculate the maximum error
def calculate_error(v, n):
    """
    Compute the maximum error for the grid.
    """
    max_error = 0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            updated_value = 0.25 * (v[i + 1, j] + v[i - 1, j] + v[i, j + 1] + v[i, j - 1])
            error = abs(updated_value - v[i, j])
            max_error = max(max_error, error)
    return max_error

# Perform checkerboard relaxation until convergence
def checkerboard_until_converged(v, n, tolerance=0.01):
    """
    Perform checkerboard relaxation until the maximum error is below the tolerance.
    """
    iterations = 0
    while True:
        max_error = calculate_error(v, n)
        if max_error < tolerance:
            break
        relax_checkerboard(v, n)
        iterations += 1
    return v, iterations

def plot_grid_size_vs_iterations(grid_sizes, tolerance=0.01):
    """
    Calculate and plot the number of iterations required for convergence
    against different grid sizes using the checkerboard relaxation method.
    """
    iterations_list = []
    for grid_size in grid_sizes:
        v = initialize_grid(grid_size)
        _, iterations = checkerboard_until_converged(v, grid_size, tolerance)
        iterations_list.append(iterations)
    
    # Plot grid size vs iterations
    plt.figure(figsize=(8, 6))
    plt.plot(grid_sizes, iterations_list, marker='o', color='b', label='Checkerboard Method')
    plt.title('Grid Size vs Number of Iterations (Checkerboard Relaxation)')
    plt.xlabel('Grid Size (n x n)')
    plt.ylabel('Number of Iterations')
    plt.grid(True)
    plt.legend()
    plt.show()

# Initialize the grid for both 9x9 and 20x20 grids
n_9x9 = 9
n_20x20 = 20

v_9x9 = initialize_grid(n_9x9)
v_20x20 = initialize_grid(n_20x20)

# Perform checkerboard relaxation for both grids
final_grid_9x9, iterations_9x9 = checkerboard_until_converged(v_9x9, n_9x9)
final_grid_20x20, iterations_20x20 = checkerboard_until_converged(v_20x20, n_20x20)

# Display the number of iterations
print(f"Iterations for 9x9 grid (Checkerboard): {iterations_9x9}")
print(f"Iterations for 20x20 grid (Checkerboard): {iterations_20x20}")

# Plot the results for visualization
def plot_equipotential(grid, title):
    plt.figure(figsize=(8, 6))
    plt.contourf(grid, levels=20, cmap='viridis')
    plt.colorbar(label='Potential (V)')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

plot_equipotential(final_grid_9x9, "Equipotential Surfaces (9x9 Grid - Checkerboard)")
plot_equipotential(final_grid_20x20, "Equipotential Surfaces (20x20 Grid - Checkerboard)")

"""grid_sizes = np.linspace(10, 300, 20, dtype=int)    # Grid sizes to test
plot_grid_size_vs_iterations(grid_sizes)"""
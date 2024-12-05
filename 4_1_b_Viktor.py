import numpy as np
import matplotlib.pyplot as plt


# Function to relax the grid
def relax(grid, grid_new, n):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            grid_new[i, j] = 0.25 * (grid[i + 1, j] + grid[i - 1, j] + grid[i, j + 1] + grid[i, j - 1])

# Function to calculate the maximum error
def calculate_error(grid, grid_new, n):
    return np.max(np.abs(grid_new[1:-1, 1:-1] - grid[1:-1, 1:-1]))

# Function to initialize the grid with the given conditions
def initialize_grid_with_center_value(n, center_value=4):
    """
    Initialize the grid for a given n, with all interior points set to 0
    except the center point, which is set to `center_value`.
    """
    # Initialize the potential grid with boundary conditions
    v = np.zeros((n + 2, n + 2))  # Include boundary points
    vnew = np.zeros_like(v)

    # Set boundary conditions
    v[0, :] = 10  # Top boundary
    v[-1, :] = 10  # Bottom boundary
    v[:, 0] = 10  # Left boundary
    v[:, -1] = 10  # Right boundary

    # Set the interior points to 0 except the center point
    center_x = (n + 1) // 2  # Center of the grid in x-direction
    center_y = (n + 1) // 2  # Center of the grid in y-direction
    v[center_x, center_y] = center_value

    return v, vnew

# Perform the relaxation method and return the final grid
def relax_until_converged(v, vnew, n, tolerance=0.01):
    iterations = 0
    max_error = tolerance + 1  # Initialize with a value greater than tolerance

    while max_error > tolerance:
        relax(v, vnew, n)
        max_error = calculate_error(v, vnew, n)
        v[1:-1, 1:-1] = vnew[1:-1, 1:-1]  # Update interior points
        iterations += 1

    return v, iterations

def plot_grid_size_vs_iterations(grid_sizes, center_values, tolerance=0.01):
    """
    Calculate and plot the number of iterations required for convergence
    against different grid sizes using the relaxation method for various center_values.
    """
    plt.figure(figsize=(12, 8))

    # Iterate over each center_value and plot
    for center_value in center_values:
        iterations_list = []

        for grid_size in grid_sizes:
            # Initialize grid with a specific center_value
            v, vnew = initialize_grid_with_center_value(grid_size, center_value=center_value)
            # Relax until converged and record the number of iterations
            _, iterations = relax_until_converged(v, vnew, grid_size, tolerance)
            iterations_list.append(iterations)

        # Plot the results for the current center_value
        plt.plot(grid_sizes, iterations_list, marker='o', label=f'Center Value {center_value}')

    # Add plot details
    plt.title('Grid Size vs Number of Iterations (Relaxation Method) with max error 1%, dt=3')
    plt.xlabel('Grid Size (n x n)')
    plt.ylabel('Number of Iterations')
    plt.grid(True)
    plt.legend()  # Add legend to distinguish between different center_values
    plt.show()

# Analyze for 9x9 grid
n_9x9 = 9
v_9x9, vnew_9x9 = initialize_grid_with_center_value(n_9x9, center_value=4)
final_grid_9x9, iterations_9x9 = relax_until_converged(v_9x9, vnew_9x9, n_9x9)

# Analyze for 20x20 grid
n_20x20 = 20
v_20x20, vnew_20x20 = initialize_grid_with_center_value(n_20x20, center_value=4)
final_grid_20x20, iterations_20x20 = relax_until_converged(v_20x20, vnew_20x20, n_20x20)

# Print the number of iterations for each grid size
print("Iterations for 9x9 grid:", iterations_9x9)
print("Iterations for 20x20 grid:", iterations_20x20)

# Example usage
grid_sizes = np.linspace(1, 150, 50, dtype=int) # Grid sizes to analyze
center_values = np.linspace(3, 5, 5)  # Center values from 3 to 5
plot_grid_size_vs_iterations(grid_sizes,center_values)
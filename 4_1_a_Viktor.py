import numpy as np
import matplotlib.pyplot as plt

# Relaxation function
def relax(grid, grid_new, n):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            grid_new[i, j] = 0.25 * (grid[i + 1, j] + grid[i - 1, j] + grid[i, j + 1] + grid[i, j - 1])

# Function to calculate the maximum error
def calculate_error(grid, grid_new, n):
    return np.max(np.abs(grid_new[1:-1, 1:-1] - grid[1:-1, 1:-1]))

# Function to initialize the grid with boundary conditions and a variable initial guess
def initialize_grid(n, initial_guess=9):
    v = np.zeros((n + 2, n + 2))  # Include boundary points
    vnew = np.zeros_like(v)

    # Set boundary conditions
    v[0, :] = 10  # Top boundary
    v[-1, :] = 10  # Bottom boundary
    v[:, 0] = 10  # Left boundary
    v[:, -1] = 10  # Right boundary

    # Initial guess for interior points
    v[1:-1, 1:-1] = initial_guess

    return v, vnew

# Function to perform relaxation until convergence
def relax_until_converged(grid, grid_new, n, tolerance=0.01):
    iterations = 0
    max_error = tolerance + 1  # Initialize with a value greater than tolerance

    while max_error > tolerance:
        relax(grid, grid_new, n)
        max_error = calculate_error(grid, grid_new, n)
        grid[1:-1, 1:-1] = grid_new[1:-1, 1:-1]  # Update interior points
        iterations += 1

    return iterations

# Function to plot grid size vs. number of iterations for different initial guesses
def plot_grid_size_vs_iterations(grid_sizes, initial_guesses, tolerance=0.01):
    plt.figure(figsize=(14, 8))

    for initial_guess in initial_guesses:
        iterations_list = []

        for grid_size in grid_sizes:
            grid, grid_new = initialize_grid(grid_size, initial_guess=initial_guess)
            iterations = relax_until_converged(grid, grid_new, grid_size, tolerance)
            iterations_list.append(iterations)

        # Plot the results for the current initial guess
        plt.plot(grid_sizes, iterations_list, marker='o', label=f'Initial Guess {initial_guess}')

    # Add plot details
    plt.title('Grid Size vs Number of Iterations (Relaxation Method) with max error 1%, dt=0.2')
    plt.xlabel('Grid Size (n x n)')
    plt.ylabel('Number of Iterations')
    plt.grid(True)
    plt.legend()  # Add legend to distinguish between different initial guesses
    plt.show()

# Define grid sizes and initial guesses to analyze
grid_sizes = np.linspace(1, 40, 200, dtype=int)  # Grid sizes from 10 to 100
initial_guesses = np.linspace(8.5, 9.5, 5)  # Initial guesses from 8.5 to 9.5 with 5 steps

plot_grid_size_vs_iterations(grid_sizes, initial_guesses)

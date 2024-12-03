import numpy as np

# Parameters for the problem
L = 10  # Length of the square region
n = 9   # Number of interior points (initially 9x9 grid)
tolerance = 0.01  # 1% accuracy

# Initialize the potential grid
v = np.zeros((n + 2, n + 2))  # Include boundary points
vnew = np.zeros_like(v)

# Set boundary conditions
v[0, :] = 10  # Top boundary
v[-1, :] = 10  # Bottom boundary
v[:, 0] = 10  # Left boundary
v[:, -1] = 10  # Right boundary

# Initial guess for interior points (10% lower than expected boundary value)
v[1:-1, 1:-1] = 9

# Relaxation function
def relax(grid, grid_new, n):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            grid_new[i, j] = 0.25 * (grid[i + 1, j] + grid[i - 1, j] + grid[i, j + 1] + grid[i, j - 1])

# Function to calculate the maximum error
def calculate_error(grid, grid_new, n):
    return np.max(np.abs(grid_new[1:-1, 1:-1] - grid[1:-1, 1:-1]))

# Perform relaxation to achieve 1% accuracy
iterations = 0
max_error = tolerance + 1  # Initialize with a value greater than tolerance

while max_error > tolerance:
    relax(v, vnew, n)
    max_error = calculate_error(v, vnew, n)
    v[1:-1, 1:-1] = vnew[1:-1, 1:-1]  # Update interior points
    iterations += 1

# Store the number of iterations for the first case
iterations_case_1 = iterations

# Repeat the process for doubled resolution (grid size increased by factor of 2)
n = 20 
v = np.zeros((n + 2, n + 2))
vnew = np.zeros_like(v)

# Set boundary conditions
v[0, :] = 10
v[-1, :] = 10
v[:, 0] = 10
v[:, -1] = 10

# Initial guess
v[1:-1, 1:-1] = 9

# Perform relaxation again
iterations = 0
max_error = tolerance + 1

while max_error > tolerance:
    relax(v, vnew, n)
    max_error = calculate_error(v, vnew, n)
    v[1:-1, 1:-1] = vnew[1:-1, 1:-1]
    iterations += 1

# Store the number of iterations for the second case
iterations_case_2 = iterations

print("Iterations for 9x9 grid:", iterations_case_1)
print("Iterations for 20x20 grid:", iterations_case_2)

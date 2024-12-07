import numpy as np
import matplotlib.pyplot as plt

# Function to perform a random walk starting at (x, y) until it hits the boundary
def random_walk(grid, start_x, start_y):
    x, y = start_x, start_y
    n, m = grid.shape
    while 0 < x < n - 1 and 0 < y < m - 1:  # Stay within the interior of the grid
        step = np.random.choice(["up", "down", "left", "right"])
        if step == "up":
            x -= 1
        elif step == "down":
            x += 1
        elif step == "left":
            y -= 1
        elif step == "right":
            y += 1
    return x, y  # Return the boundary point reached

# Function to compute the Green's function for an interior point
def compute_greens_function(grid, start_x, start_y, num_walks):
    n, m = grid.shape
    G = np.zeros_like(grid)  # Initialize the Green's function matrix
    for _ in range(num_walks):
        boundary_x, boundary_y = random_walk(grid, start_x, start_y)
        G[boundary_x, boundary_y] += 1  # Increment count at the boundary point
    return G / np.sum(G)  # Normalize so the total probability is 1

# Initialize the grid with asymmetric boundary conditions
def initialize_asymmetric_boundaries(n):
    """
    Initialize the grid with specific boundary conditions:
    - Top and bottom sides: 10V
    - Left and right sides: 5V
    """
    grid = np.zeros((n + 2, n + 2))  # Include boundary points
    grid[0, :] = 10  # Top boundary
    grid[-1, :] = 10  # Bottom boundary
    grid[:, 0] = 5   # Left boundary
    grid[:, -1] = 5  # Right boundary
    return grid

# Function to plot the potential with boundaries
def plot_potential_with_boundaries(grid, potential, title, titlefontsize=14, colorbarfont=16, labelfontsize=14):
    """
    Plot the potential with boundary values explicitly included.
    """
    combined_grid = potential.copy()
    combined_grid[0, :] = grid[0, :]  # Top boundary
    combined_grid[-1, :] = grid[-1, :]  # Bottom boundary
    combined_grid[:, 0] = grid[:, 0]  # Left boundary
    combined_grid[:, -1] = grid[:, -1]  # Right boundary

    # Plot the combined grid
    plt.figure(figsize=(8, 6))
    plt.contourf(combined_grid, levels=20, cmap='viridis')
    plt.colorbar(label="Potential (V)")
    plt.title(title, fontsize=titlefontsize)
    plt.xlabel("X", fontsize=labelfontsize)
    plt.ylabel("Y", fontsize=labelfontsize)
    plt.show()

# Main script
if __name__ == "__main__":
    n = 9  # Grid size (excluding boundaries)
    num_walks = 200  # Number of walkers per interior point

    # Initialize grid
    grid = initialize_asymmetric_boundaries(n)

    # Compute potential matrix for interior points
    potential = np.zeros_like(grid)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            G = compute_greens_function(grid, i, j, num_walks)
            # Compute the potential as a weighted sum of boundary probabilities
            potential[i, j] = (
                np.sum(G[0, :]) * 10 +  # Top boundary
                np.sum(G[-1, :]) * 10 +  # Bottom boundary
                np.sum(G[:, 0]) * 5 +   # Left boundary
                np.sum(G[:, -1]) * 5    # Right boundary
            )
    # Plot the potential with boundaries
    title = f'Potential from G, grid size = {n}, n-walks = {num_walks}'
    plot_potential_with_boundaries(grid, potential, title)


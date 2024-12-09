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

# Function to plot Green's function for a specific interior point
def plot_greens_function(G, title, target_point, titlefontsize=14, colorbarfont=16, labelfontsize=14):
    """
    Plot the Green's function (normalized probabilities) for boundary points.
    """
    plt.figure(figsize=(8, 6))
    plt.contourf(G, levels=20, cmap='viridis')
    plt.colorbar(label="Green's Function (Probability)")
    plt.title(f"{title} for Point {target_point}", fontsize=titlefontsize)
    plt.xlabel("X", fontsize=labelfontsize)
    plt.ylabel("Y", fontsize=labelfontsize)
    plt.show()

# Main script
if __name__ == "__main__":
    n = 9  # Grid size (excluding boundaries)
    num_walks = 200  # Number of walkers per interior point

    # Initialize grid
    grid = initialize_asymmetric_boundaries(n)

    # Compute Green's function for a single interior point
    target_point = (5, 5)  # Interior point
    G = compute_greens_function(grid, target_point[0], target_point[1], num_walks)

    # Plot the Green's function for the selected point
    plot_greens_function(G, title=f"Green's Function", target_point=target_point)

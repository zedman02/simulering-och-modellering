import numpy as np
import random
import matplotlib.pyplot as plt

def random_walk_a(steps):
    # Starting position
    x, y = 0, 0
    x_positions = [x]
    y_positions = [y]

    # Generate the random walk
    for _ in range(steps):
        direction = int(random.random() * 4)  # Randomly pick a direction (0, 1, 2, or 3)
        if direction == 0:
            x += 1  # Move right
        elif direction == 1:
            x -= 1  # Move left
        elif direction == 2:
            y += 1  # Move up
        elif direction == 3:
            y -= 1  # Move down
        x_positions.append(x)
        y_positions.append(y)
    
    return x_positions, y_positions

def avoiding_walk_2D(N, max_attempts):
    """Generate a self-avoiding walk and return its RMS values for successful walks."""
    walk_distances = []  # List to store end-to-end distances for successful walks

    for _ in range(max_attempts):
        walk = [(0, 0)]
        visited = set(walk)
        success = True

        for _ in range(N):
            x, y = walk[-1]
            possible_moves = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            moves = [move for move in possible_moves if move not in visited]

            if not moves:  # No valid moves, walk fails
                success = False
                break

            next_move = random.choice(moves)
            walk.append(next_move)
            visited.add(next_move)

        if success:
            # Calculate end-to-end distance for the successful walk
            x_final, y_final = walk[-1]
            distance = np.sqrt(x_final**2 + y_final**2)
            walk_distances.append(distance)

    if walk_distances:
        rms_distance = np.sqrt(np.mean(np.square(walk_distances)))
    else:
        rms_distance = 0

    return rms_distance

def rmsf_random_walk(num_walks, step_counts):
    """Calculate RMS values for all random walks."""
    rms_distances = []

    for steps in step_counts:
        squared_distances = []

        for _ in range(num_walks):
            x, y = random_walk_a(steps)
            x_final, y_final = x[-1], y[-1]
            distance = np.sqrt(x_final**2 + y_final**2)
            squared_distances.append(distance**2)

        mean_squared_distance = np.mean(squared_distances)
        rms_distance = np.sqrt(mean_squared_distance)
        rms_distances.append(rms_distance)

    return step_counts, rms_distances

def plot_self_avoiding_vs_random():
    num_walks = 100
    max_attempts = 100
    step_counts = list(range(1, 31))

    # Calculate RMS for self-avoiding walks
    rms_self_avoiding = [avoiding_walk_2D(steps, max_attempts) for steps in step_counts]

    # Calculate RMS for random walks
    steps, rms_random_walk = rmsf_random_walk(num_walks, step_counts)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(step_counts, rms_self_avoiding, label="Self-Avoiding Walks (Successful Only)", marker='o')
    plt.plot(steps, rms_random_walk, label="Random Walks", marker='o')
    plt.title("RMS Distance vs Steps")
    plt.xlabel("Number of Steps")
    plt.ylabel("RMS Distance")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the plot function
plot_self_avoiding_vs_random()

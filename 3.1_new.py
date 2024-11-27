import numpy as np
import matplotlib.pyplot as plt


def metropolis(delta, n_steps, burn_in):
    samples = []
    x = 0.0  # Initial position
    accepted = 0  # Counter for accepted moves

    for step in range(n_steps):
        x_new = x + np.random.uniform(-delta, delta)

        # Compute the acceptance probability
        if x_new < 0:
            p_accept = 0
        else:
            p_accept = min(1, np.exp(-x_new) / np.exp(-x))

        if np.random.random() < p_accept:
            x = x_new
            accepted += 1

        if step >= burn_in:
            samples.append(x)

    acceptance_rate = accepted / n_steps
    return samples, acceptance_rate


def calculate_independent_error(samples):
    var = np.var(samples)
    N = len(samples)
    return var / N


def calculate_rms_error(samples):
    return (np.mean(samples) - 1) ** 2


# Main simulation parameters
n_steps = 10000  # Total steps
burn_in = 500  # Burn-in steps
delta_values = np.linspace(0.01, 10, 50)  # Range of delta
runs = 10

# Storage for results
independent_errors = []
rms_errors = []
acceptance_rates = []

for delta in delta_values:
    temp_errors = []
    temp_rms = []
    temp_acceptance = []

    for _ in range(runs):
        samples, acceptance_rate = metropolis(delta, n_steps, burn_in)
        temp_errors.append(calculate_independent_error(samples))
        temp_rms.append(calculate_rms_error(samples))
        temp_acceptance.append(acceptance_rate)

    acceptance_rates.append(np.mean(temp_acceptance))
    rms_errors.append(np.sqrt(np.mean(temp_rms)))
    independent_errors.append(np.sqrt(np.mean(temp_errors)))

# Plotting results
plt.figure(figsize=(10, 6))

# Standard error vs delta
plt.subplot(2, 1, 1)
plt.plot(delta_values, independent_errors, label='Error estimate assuming independent points')
plt.plot(delta_values, rms_errors, label='RMS Error')
plt.xlabel('Delta')
plt.ylabel('Error')
plt.legend()
plt.title(f'Different Errors vs Delta averaged over {runs} MC runs (N0 = {burn_in}, steps = {n_steps})')

# Acceptance rate vs delta
plt.subplot(2, 1, 2)
plt.plot(delta_values, acceptance_rates, label='Acceptance Rate', color='green')
plt.xlabel('Delta')
plt.ylabel('Acceptance Rate')
plt.legend()
plt.title('Average Acceptance Rate vs Delta')

plt.tight_layout()
plt.show()
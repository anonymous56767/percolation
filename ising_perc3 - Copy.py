import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque

# Parameters
L = 50
n_steps = 300
equilibrium_cutoff = 50

# Affinity matrix (symmetric)
affinity_matrix = np.array([
    [ 0.8, -0.2,  0.2,  0.3, -0.1],
    [-0.2,  0.8,  0.4, -0.2,  0.1],
    [ 0.2,  0.4,  0.8, -0.3,  0.2],
    [ 0.3, -0.2, -0.3,  0.6,  0.5],
    [-0.1,  0.1,  0.2,  0.5,  0.9],
])
n_colors = affinity_matrix.shape[0]
neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def calculate_total_energy(grid):
    energy = 0.0
    for i in range(L):
        for j in range(L):
            state = grid[i, j]
            for dx, dy in neighbors:
                ni, nj = (i + dx) % L, (j + dy) % L
                neighbor_state = grid[ni, nj]
                energy += affinity_matrix[state, neighbor_state]
    return -0.5 * energy

def metropolis_step(grid, temperature):
    i, j = np.random.randint(0, L), np.random.randint(0, L)
    current_state = grid[i, j]
    new_state = np.random.randint(0, n_colors)
    if new_state == current_state:
        return 0
    dE = sum(
        affinity_matrix[new_state, grid[(i + dx) % L, (j + dy) % L]] -
        affinity_matrix[current_state, grid[(i + dx) % L, (j + dy) % L]]
        for dx, dy in neighbors
    )
    if dE < 0 or np.random.rand() < np.exp(-dE / temperature):
        grid[i, j] = new_state
        return dE
    return 0

def find_largest_cluster_fraction(grid):
    visited = np.zeros_like(grid, dtype=bool)
    max_cluster_size = 0
    for i in range(L):
        for j in range(L):
            if not visited[i, j]:
                state = grid[i, j]
                cluster_size = 0
                queue = deque([(i, j)])
                while queue:
                    x, y = queue.popleft()
                    if visited[x, y] or grid[x, y] != state:
                        continue
                    visited[x, y] = True
                    cluster_size += 1
                    for dx, dy in neighbors:
                        nx, ny = (x + dx) % L, (y + dy) % L
                        if not visited[nx, ny] and grid[nx, ny] == state:
                            queue.append((nx, ny))
                max_cluster_size = max(max_cluster_size, cluster_size)
    return max_cluster_size / (L * L)

# Continuous range of temperatures
temperatures = np.linspace(0.1, 0.9, 9)  # 50 points from 0.5 to 0.9
average_energies = []
largest_cluster_fractions = []

for T in temperatures:
    print(f"\nSimulating T = {T}")
    grid = np.random.randint(0, n_colors, size=(L, L))
    energies = []
    average_grid = np.zeros((L, L), dtype=float)

    for step in tqdm(range(n_steps), desc=f"T = {T}"):
        for _ in range(L * L):
            metropolis_step(grid, T)
        energy = calculate_total_energy(grid)
        energies.append(energy)
        average_grid += grid

    avg_energy = np.mean(energies[equilibrium_cutoff:])
    average_energies.append(avg_energy)

    # Use rounded time-averaged grid for cluster calc
    avg_grid = np.rint(average_grid / n_steps).astype(int)
    frac = find_largest_cluster_fraction(avg_grid)
    largest_cluster_fractions.append(frac)

# Plot: Average Energy vs Temperature
plt.figure()
plt.plot(temperatures, average_energies, marker='o', linestyle='-', color='blue')
plt.xlabel("Temperature")
plt.ylabel("Average Energy")
plt.title("Average Energy vs Temperature")
plt.grid(True)
plt.show()

# Plot: Largest Cluster Fraction vs Temperature
plt.figure()
plt.plot(temperatures, largest_cluster_fractions, marker='o', linestyle='-', color='green')
plt.xlabel("Temperature")
plt.ylabel("Fraction of Largest Cluster")
plt.title("Largest Cluster Fraction vs Temperature")
plt.grid(True)
plt.show()

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# Define logistic function
def logistic(T, L, k, T0, b):
    return L / (1 + np.exp(-k * (T - T0))) + b

# Fit logistic curve
popt, _ = curve_fit(logistic, temperatures, largest_cluster_fractions, 
                    p0=[1, 10, 0.7, 0], maxfev=10000)

L_fit, k_fit, T0_fit, b_fit = popt
fit_temperatures = np.linspace(min(temperatures), max(temperatures), 300)
fit_fractions = logistic(fit_temperatures, *popt)

# Plot original and fitted curves
plt.figure()
plt.plot(temperatures, largest_cluster_fractions, 'o', label='Data', color='green')
plt.plot(fit_temperatures, fit_fractions, '-', label='Logistic Fit', color='black')
plt.axvline(T0_fit, linestyle='--', color='red', label=f'Max Change at T={T0_fit:.3f}')
plt.xlabel("Temperature")
plt.ylabel("Fraction of Largest Cluster")
plt.title("Largest Cluster Fraction vs Temperature")
plt.legend()
plt.grid(True)
plt.show()

# Print the inflection point temperature
print(f"🔍 Temperature of max change (inflection point): T = {T0_fit:.4f}")

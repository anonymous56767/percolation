import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
from scipy.optimize import curve_fit

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

def metropolis_step(grid, temperature):
    i, j = np.random.randint(0, L), np.random.randint(0, L)
    current_state = grid[i, j]
    new_state = np.random.randint(0, n_colors)
    if new_state == current_state:
        return
    dE = sum(
        affinity_matrix[new_state, grid[(i + dx) % L, (j + dy) % L]] -
        affinity_matrix[current_state, grid[(i + dx) % L, (j + dy) % L]]
        for dx, dy in neighbors
    )
    if dE < 0 or np.random.rand() < np.exp(-dE / temperature):
        grid[i, j] = new_state

def find_largest_cluster_fraction(grid):
    visited = np.zeros_like(grid, dtype=bool)
    max_cluster_size = 0
    for i in range(L):
        for j in range(L):
            if not visited[i, j]:
                state = grid[i, j]
                queue = deque([(i, j)])
                cluster_size = 0
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

# Simulation
temperatures = np.linspace(0.1, 0.9, 60)
largest_cluster_fractions = []

for T in tqdm(temperatures, desc="Simulating temperatures"):
    grid = np.random.randint(0, n_colors, size=(L, L))
    avg_grid = np.zeros_like(grid, dtype=float)

    for step in range(n_steps):
        for _ in range(L * L):
            metropolis_step(grid, T)
        avg_grid += grid

    rounded_avg_grid = np.rint(avg_grid / n_steps).astype(int)
    frac = find_largest_cluster_fraction(rounded_avg_grid)
    largest_cluster_fractions.append(frac)

# Logistic fit
def logistic(T, L, k, T0, b):
    return L / (1 + np.exp(-k * (T - T0))) + b

popt, _ = curve_fit(logistic, temperatures, largest_cluster_fractions, 
                    p0=[1, 10, 0.5, 0], maxfev=10000)
L_fit, k_fit, T0_fit, b_fit = popt
T_dense = np.linspace(min(temperatures), max(temperatures), 300)
fit_curve = logistic(T_dense, *popt)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(temperatures, largest_cluster_fractions, 'o', label='Simulated Data', color='darkgreen')
plt.plot(T_dense, fit_curve, '-', label='Logistic Fit', color='black', linewidth=2)
plt.axvline(T0_fit, linestyle='--', color='crimson', label=f'Inflection Point: T = {T0_fit:.3f}')
plt.xlabel("Temperature", fontsize=12)
plt.ylabel("Fraction of Largest Cluster", fontsize=12)
plt.title("Cluster Growth vs Temperature", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Output the inflection point
print(f"🔍 Inflection point (maximum change): T = {T0_fit:.4f}")

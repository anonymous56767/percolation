import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parameters
L = 50
n_steps = 300
equilibrium_cutoff = 50
temperatures = np.linspace(0.1, 0.8 , 8)  # Continuous temp range

# Manually define a symmetric affinity matrix
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

def metropolis_step(grid, T):
    i, j = np.random.randint(0, L), np.random.randint(0, L)
    current = grid[i, j]
    proposal = np.random.randint(0, n_colors)
    if proposal == current:
        return 0
    dE = 0.0
    for dx, dy in neighbors:
        ni, nj = (i + dx) % L, (j + dy) % L
        neighbor = grid[ni, nj]
        dE += affinity_matrix[proposal, neighbor] - affinity_matrix[current, neighbor]
    if dE < 0 or np.random.rand() < np.exp(-dE / T):
        grid[i, j] = proposal
        return dE
    return 0

# Run simulation
avg_energies = []

for T in tqdm(temperatures, desc="Sweeping temperatures"):
    grid = np.random.randint(0, n_colors, size=(L, L))
    energies = []

    for step in range(n_steps):
        for _ in range(L * L):
            metropolis_step(grid, T)
        energy = calculate_total_energy(grid)
        energies.append(energy)

    avg_energies.append(np.mean(energies[equilibrium_cutoff:]))

# Plot result
plt.figure(figsize=(8, 5))
plt.plot(temperatures, avg_energies, marker='o', color='navy')
plt.xlabel("Temperature")
plt.ylabel("Average Energy")
plt.title("Average Energy vs Temperature (Continuous)")
plt.grid(True)
plt.tight_layout()
plt.show()

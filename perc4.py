import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# Parameters
L = 50              # Grid size
n_steps = 300       # Monte Carlo steps per temperature
equilibrium_cutoff = 50  # Steps to exclude from energy average (transient)

# Manually define a symmetric affinity matrix
affinity_matrix = np.array([
    [ 0.8, -0.2,  0.2,  0.3, -0.1],
    [-0.2,  0.8,  0.4, -0.2,  0.1],
    [ 0.2,  0.4,  0.8, -0.3,  0.2],
    [ 0.3, -0.2, -0.3,  0.6,  0.5],
    [-0.1,  0.1,  0.2,  0.5,  0.9],
])

n_colors = affinity_matrix.shape[0]

# Colormap
cmap = plt.get_cmap('viridis', n_colors)

# 4-neighbor model
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

    dE = 0.0
    for dx, dy in neighbors:
        ni, nj = (i + dx) % L, (j + dy) % L
        neighbor_state = grid[ni, nj]
        dE += affinity_matrix[new_state, neighbor_state] - affinity_matrix[current_state, neighbor_state]

    if dE < 0 or np.random.rand() < np.exp(-dE / temperature):
        grid[i, j] = new_state
        return dE
    return 0

# -----------------------------
# 🎞️ Animation at T = 1.0
# -----------------------------
temperature = 1.0
grid = np.random.randint(0, n_colors, size=(L, L))
frames = []
energy_list = []
average_grid = np.zeros((L, L), dtype=float)

for step in tqdm(range(n_steps), desc="Animating T=1.0"):
    for _ in range(L * L):
        metropolis_step(grid, temperature)

    energy = calculate_total_energy(grid)
    energy_list.append(energy)
    average_grid += grid

    if step % 5 == 0:
        frames.append(grid.copy())

time_averaged_grid = average_grid / n_steps

# 🎞️ Animation
fig, ax = plt.subplots()
im = ax.imshow(frames[0], cmap=cmap, vmin=0, vmax=n_colors - 1)
fig.colorbar(im, ax=ax, ticks=range(n_colors))
plt.title("Ising-like Model (T=1.0)")

def update(frame):
    im.set_data(frame)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
plt.show()

# 📉 Energy vs Time
plt.figure()
plt.plot(range(n_steps), energy_list, color='darkred')
plt.xlabel("Time Step")
plt.ylabel("Total Energy")
plt.title("Energy vs Time (T=1.0)")
plt.grid(True)
plt.show()

# 🌈 Time-Averaged Grid at T=1.0
plt.figure(figsize=(6, 6))
plt.imshow(time_averaged_grid, cmap=cmap, vmin=0, vmax=n_colors - 1)
plt.colorbar(ticks=np.arange(n_colors))
plt.title("Time-Averaged Color at Each (x, y) (T=1.0)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# ---------------------------------------
# 🔥 Multiple Temperatures Simulation
# ---------------------------------------
temperatures = [0.4,0.5,0.6,0.7]
average_grids_temp = []
average_energies = []

for T in temperatures:
    print(f"\nSimulating for T = {T}")
    grid = np.random.randint(0, n_colors, size=(L, L))
    average_grid = np.zeros((L, L), dtype=float)
    energies = []

    for step in tqdm(range(n_steps), desc=f"T = {T}"):
        for _ in range(L * L):
            metropolis_step(grid, T)
        energy = calculate_total_energy(grid)
        energies.append(energy)
        average_grid += grid

    avg_grid = average_grid / n_steps
    average_grids_temp.append((T, avg_grid))
    
    # Energy average over the last part (after equilibrium)
    stable_energy = np.mean(energies[equilibrium_cutoff:])
    average_energies.append(stable_energy)

# 📊 Time-Averaged Grid per Temperature
fig, axs = plt.subplots(1, len(temperatures), figsize=(4 * len(temperatures), 4))
for idx, (T, avg_grid) in enumerate(average_grids_temp):
    ax = axs[idx]
    im = ax.imshow(avg_grid, cmap=cmap, vmin=0, vmax=n_colors - 1)
    ax.set_title(f"T = {T}")
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("Time-Averaged Color State vs Temperature", fontsize=16)
plt.tight_layout()
plt.show()

# 📈 Energy vs Temperature
plt.figure()
plt.plot(temperatures, average_energies, marker='o', linestyle='-', color='blue')
plt.xlabel("Temperature")
plt.ylabel("Average Energy")
plt.title("Average Energy vs Temperature")
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import Voronoi
from scipy.optimize import curve_fit
from scipy.sparse.csgraph import connected_components

# === POISSON DISC SAMPLING ===
def poisson_disc_samples(width, height, r, k=30):
    import random
    from math import sqrt, cos, sin, pi
    cell_size = r / sqrt(2)
    grid_width = int(width / cell_size) + 1
    grid_height = int(height / cell_size) + 1
    grid = [None] * (grid_width * grid_height)

    def grid_coords(p):
        return int(p[0] / cell_size), int(p[1] / cell_size)

    def fits(p):
        gx, gy = grid_coords(p)
        for i in range(max(0, gx - 2), min(grid_width, gx + 3)):
            for j in range(max(0, gy - 2), min(grid_height, gy + 3)):
                neighbor = grid[i + j * grid_width]
                if neighbor:
                    dx = neighbor[0] - p[0]
                    dy = neighbor[1] - p[1]
                    if dx * dx + dy * dy < r * r:
                        return False
        return True

    def add_point(p):
        points.append(p)
        active.append(p)
        gx, gy = grid_coords(p)
        grid[gx + gy * grid_width] = p

    points = []
    active = []
    p0 = (random.uniform(0, width), random.uniform(0, height))
    add_point(p0)

    while active:
        idx = random.randint(0, len(active) - 1)
        center = active[idx]
        found = False
        for _ in range(k):
            theta = random.uniform(0, 2 * pi)
            rad = random.uniform(r, 2 * r)
            px = center[0] + rad * cos(theta)
            py = center[1] + rad * sin(theta)
            if 0 <= px < width and 0 <= py < height:
                if fits((px, py)):
                    add_point((px, py))
                    found = True
        if not found:
            active.pop(idx)
    return np.array(points)

# === CORE FUNCTIONS ===
def logistic(p, k, p_c):
    return 1 / (1 + np.exp(-k * (p - p_c)))

def build_adjacency(vor, ridge_mask):
    n = len(vor.vertices)
    adj = np.zeros((n, n))
    for i, ridge in enumerate(vor.ridge_vertices):
        if -1 in ridge or not ridge_mask[i]:
            continue
        u, v = ridge
        adj[u, v] = 1
        adj[v, u] = 1
    return adj

def largest_cluster_fraction(vor, ridge_probs, p):
    mask = ridge_probs < p
    adj = build_adjacency(vor, mask)
    n_components, labels = connected_components(adj)
    if len(labels) == 0:
        return 0
    largest = np.max(np.bincount(labels))
    return largest / len(vor.vertices)

# === ANIMATION (FIXED DENSITY) ===
np.random.seed(42)
n_points = 150
min_dist = 1 / np.sqrt(n_points)
points = poisson_disc_samples(1, 1, r=min_dist)
vor = Voronoi(points)
ridge_probs = np.random.rand(len(vor.ridge_vertices))
p_vals = np.linspace(0, 1, 100)
fractions = []

fig, (ax_vor, ax_plot) = plt.subplots(1, 2, figsize=(14, 6))

def update(frame):
    ax_vor.clear()
    ax_plot.clear()

    p = p_vals[frame]
    mask = ridge_probs < p

    for i, ridge in enumerate(vor.ridge_vertices):
        if -1 in ridge:
            continue
        u, v = ridge
        color = 'red' if mask[i] else 'gray'
        lw = 2 if mask[i] else 0.5
        alpha = 1 if mask[i] else 0.2
        ax_vor.plot([vor.vertices[u][0], vor.vertices[v][0]],
                    [vor.vertices[u][1], vor.vertices[v][1]],
                    color=color, lw=lw, alpha=alpha)

    ax_vor.scatter(points[:, 0], points[:, 1], s=10, color='blue')
    ax_vor.set_xlim(0, 1)
    ax_vor.set_ylim(0, 1)
    ax_vor.set_xticks([])
    ax_vor.set_yticks([])
    ax_vor.set_title(f"Voronoi Percolation at p = {p:.2f}")

    frac = largest_cluster_fraction(vor, ridge_probs, p)
    if len(fractions) <= frame:
        fractions.append(frac)

    ax_plot.plot(p_vals[:frame+1], fractions[:frame+1], label="Largest Cluster Fraction", color='blue')

    try:
        popt, _ = curve_fit(logistic, p_vals[:frame+1], fractions[:frame+1], p0=[10, 0.5], maxfev=5000)
        k_fit, thres_p = popt
        fitted = logistic(p_vals, k_fit, thres_p)
        ax_plot.plot(p_vals, fitted, '--', color='green', label='Logistic Fit')
        ax_plot.axvline(x=thres_p, color='orange', linestyle='--', label=f"$p_c$ ≈ {thres_p:.3f}")
    except:
        pass

    ax_plot.set_xlim(0, 1)
    ax_plot.set_ylim(0, 1)
    ax_plot.set_xlabel("Percolation Probability p")
    ax_plot.set_ylabel("Largest Connected Cluster Fraction")
    ax_plot.set_title("Percolation Curve")
    ax_plot.grid(True)
    ax_plot.legend()

ani = animation.FuncAnimation(fig, update, frames=len(p_vals), interval=200, repeat=False)
plt.tight_layout()
plt.show()

# === PC vs DENSITY AFTER ANIMATION ===
densities = np.linspace(100, 1000, 100, dtype=int)
pc_list = []

for n_points in densities:
    min_dist = 1 / np.sqrt(n_points)
    points = poisson_disc_samples(1, 1, r=min_dist)
    if len(points) < 4:
        continue
    vor = Voronoi(points)
    ridge_probs = np.random.rand(len(vor.ridge_vertices))
    fractions = [largest_cluster_fraction(vor, ridge_probs, p) for p in p_vals]

    try:
        popt, _ = curve_fit(logistic, p_vals, fractions, p0=[10, 0.5], maxfev=5000)
        p_c = popt[1]
    except:
        p_c = p_vals[np.argmax(np.gradient(fractions))]

    pc_list.append(p_c)

# === PLOT p_c vs DENSITY ===
plt.figure(figsize=(8, 6))
plt.plot(densities[:len(pc_list)], pc_list, 'o-', color='purple', label='Estimated $p_c$')
plt.xlabel("Nucleation Point Count (Density)")
plt.ylabel("Percolation Threshold $p_c$")
plt.title("Percolation Threshold vs Nucleation Density")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === AVERAGE p_c ===
avg_pc = np.mean(pc_list)
print(f"Average p_c over densities: {avg_pc:.4f}")

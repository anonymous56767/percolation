import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components

# === PARAMETERS ===
np.random.seed(42)
n_points = 150
d_cutoff = 1.5     # Max distance for edge consideration
d_scale = 0.05      # Controls how fast connection probability decays with distance
p_vals = np.linspace(0, 1, 100)

# === POINT GENERATION ===
points = np.random.rand(n_points, 2)
dist_matrix = squareform(pdist(points))

# === EDGE PROBABILITIES BASED ON DISTANCE ===
prob_matrix = np.exp(-dist_matrix / d_scale)
prob_matrix[dist_matrix > d_cutoff] = 0  # Enforce hard cutoff

# Random static values to compare against
rand_matrix = np.random.rand(n_points, n_points)
rand_matrix = np.triu(rand_matrix, 1)

# === LOGISTIC ===
def logistic(p, k, p_c):
    return 1 / (1 + np.exp(-k * (p - p_c)))

# === BUILD ADJACENCY FOR A GIVEN p ===
def build_adjacency(p):
    threshold = prob_matrix * p
    adj = (rand_matrix < threshold).astype(int)
    adj = np.triu(adj, 1)
    adj += adj.T
    return adj

# === LARGEST CLUSTER FRACTION ===
def largest_cluster_fraction(p):
    adj = build_adjacency(p)
    n_components, labels = connected_components(adj)
    if len(labels) == 0:
        return 0
    largest = np.max(np.bincount(labels))
    return largest / n_points

# === ANIMATION ===
fractions = []
fig, (ax_graph, ax_plot) = plt.subplots(1, 2, figsize=(14, 6))

def update(frame):
    ax_graph.clear()
    ax_plot.clear()

    p = p_vals[frame]
    adj = build_adjacency(p)

    # Draw edges
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if adj[i, j]:
                ax_graph.plot([points[i, 0], points[j, 0]],
                              [points[i, 1], points[j, 1]],
                              color='red', alpha=0.6, linewidth=0.8)

    # Draw nodes
    ax_graph.scatter(points[:, 0], points[:, 1], s=10, color='blue')
    ax_graph.set_xlim(0, 1)
    ax_graph.set_ylim(0, 1)
    ax_graph.set_xticks([])
    ax_graph.set_yticks([])
    ax_graph.set_title(f"Graph at p = {p:.2f}")

    # Percolation tracking
    frac = largest_cluster_fraction(p)
    if len(fractions) <= frame:
        fractions.append(frac)

    ax_plot.plot(p_vals[:frame+1], fractions[:frame+1], label="Largest Cluster Fraction", color='blue')

    try:
        popt, _ = curve_fit(logistic, p_vals[:frame+1], fractions[:frame+1], p0=[10, 0.5], maxfev=5000)
        k_fit, thres_p = popt
        fitted = logistic(p_vals, *popt)
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

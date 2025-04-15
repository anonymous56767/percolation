# perc1
 test percolation repo
This Python script simulates and visualizes a percolation process using Voronoi diagrams and logistic curves. Below is a breakdown of the code:

---

### *Key Concepts in the Code*
1. **Poisson Disc Sampling (poisson_disc_samples function):**
   - Generates a set of points in a 2D space while maintaining a minimum distance (r) between points.
   - Used to create evenly distributed nucleation points for the Voronoi diagram.

2. *Voronoi Diagram:*
   - Generated using the scipy.spatial.Voronoi class to partition the 2D space into regions based on the distance to nucleation points.
   - Ridge connections between points are used for simulating percolation.

3. *Percolation Simulation:*
   - Assigns a random probability to each ridge (connection between Voronoi vertices).
   - Simulates the formation of clusters by progressively increasing a threshold p and identifying connected components using an adjacency matrix.

4. *Logistic Curve Fitting:*
   - Fits a logistic function to the largest connected cluster's fraction as a function of p.
   - Identifies the percolation threshold (p_c), where a large connected cluster emerges.

5. *Animation:*
   - Visualizes the percolation process step-by-step as p increases.
   - Shows the Voronoi diagram with active (percolating) ridges and plots the largest cluster fraction over time.

6. *Analysis of Percolation Threshold vs Density:*
   - Varies the density of nucleation points.
   - Calculates the percolation threshold (p_c) for each density and plots the relationship.

---

### *Detailed Code Walkthrough*
1. *Imports (Lines 1–6):*
   - Libraries for numerical calculations (numpy), plotting (matplotlib), Voronoi generation (scipy.spatial.Voronoi), and curve fitting (scipy.optimize.curve_fit).

2. *Poisson Disc Sampling (Lines 9–58):*
   - Generates points in a 2D space ensuring no two points are closer than a specified radius (r).

3. *Core Functions (Lines 60–82):*
   - logistic: Defines a logistic function for curve fitting.
   - build_adjacency: Constructs an adjacency matrix for Voronoi vertices based on active ridges.
   - largest_cluster_fraction: Computes the fraction of points in the largest connected cluster for a given threshold p.

4. *Animation of Percolation (Lines 84–146):*
   - Creates an animation showing the evolution of percolation.
   - Displays the active ridges in the Voronoi diagram and the percolation curve (largest cluster fraction vs. p).

5. *Percolation Threshold vs Density (Lines 148–178):*
   - Simulates percolation for varying densities of nucleation points.
   - Fits a logistic curve to estimate the percolation threshold (p_c) for each density.

6. *Average Percolation Threshold (Lines 180–182):*
   - Computes and prints the average percolation threshold (p_c) across all densities.

---

### *Outputs*
1. *Animation:*
   - Shows the percolation process for a fixed density of points.
   - Includes a plot of the largest cluster fraction (f) vs. percolation probability (p).

2. *Percolation Threshold Analysis:*
   - A plot showing how the percolation threshold (p_c) varies with nucleation density.

3. *Average Threshold:*
   - Prints the average percolation threshold across densities.

---

### *Applications*
- Percolation theory is widely used in physics, materials science, and network theory to study phase transitions and connectivity.
- This script provides insights into how connectivity emerges in random networks as a function of density and connection probability.

Let me know if you need further clarification or detailed explanations of specific parts!

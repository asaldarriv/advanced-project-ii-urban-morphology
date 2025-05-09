"""
Urban Network Entropy Simulation

This script simulates how orientation entropy evolves in a grid-based street network
as the node positions are perturbed using truncated normal distributions.
It computes entropy over a range of perturbation levels (sigma values), allows parallel processing
to improve performance, saves results to an Excel file, and generates a boxplot for analysis.
"""

# --- Standard Libraries ---
import os
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

# --- Third-party Libraries ---
import numpy as np
import osmnx as ox
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from shapely.geometry import LineString


# ----------------------------
# STEP 1: GRAPH INITIALIZATION
# ----------------------------

def create_base_graph(n_rows=10, n_cols=10):
    """
    Create a 2D grid graph and convert it to a MultiGraph with geographic metadata.
    """
    G = nx.grid_2d_graph(n_rows, n_cols)
    G = nx.MultiGraph(G)
    G.graph["crs"] = "EPSG:4326"
    return G


def center_node_positions(G):
    """
    Center node positions around (0, 0) in Cartesian space.
    """
    n_cols = max(x for _, x in G.nodes()) + 1
    n_rows = max(y for y, _ in G.nodes()) + 1
    center_x, center_y = (n_cols - 1) / 2, -(n_rows - 1) / 2
    return {node: (float(node[1]) - center_x, float(-node[0]) - center_y) for node in G.nodes()}


def assign_node_attributes(G, positions):
    """
    Assign x and y positional attributes to each node.
    """
    nx.set_node_attributes(G, {node: x for node, (x, _) in positions.items()}, "x")
    nx.set_node_attributes(G, {node: y for node, (_, y) in positions.items()}, "y")
    return G


def calculate_edge_attributes(G, pos):
    """
    Compute length, geometry, and bearing for each edge in the graph.
    """
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        length = np.hypot(x2 - x1, y2 - y1)
        geom = LineString([(x1, y1), (x2, y2)])
        bearing = ox.bearing.calculate_bearing(y1, x1, y2, x2)
        G[u][v][0]["length"] = float(length)
        G[u][v][0]["geometry"] = geom
        G[u][v][0]["bearing"] = float(bearing)
    return G


# -------------------------------
# STEP 2: PLOTTING INITIAL STATE
# -------------------------------

def plot_graph(G, pos, title, filename):
    """
    Save a plot of the graph showing edges and nodes.
    """
    fig, ax = plt.subplots()
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color='black')
    x_vals = [x for x, y in pos.values()]
    y_vals = [y for x, y in pos.values()]
    ax.scatter(x_vals, y_vals, color='red')
    ax.set_aspect('equal')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    plt.savefig(filename)
    plt.close()


# -----------------------------------
# STEP 3: SIMULATION IMPLEMENTATION
# -----------------------------------

def perturb_graph(G, pos, sigma):
    """
    Apply a truncated normal perturbation to each node position.
    """
    a, b = -3, 3
    return {
        node: (
            x + truncnorm.rvs(a, b, loc=0, scale=sigma),
            y + truncnorm.rvs(a, b, loc=0, scale=sigma)
        )
        for node, (x, y) in pos.items()
    }


def run_simulation(args):
    """
    Run a single entropy calculation for a perturbed version of the graph.
    """
    sigma, rep, G, initial_position = args
    G_copy = deepcopy(G)
    perturbed_position = perturb_graph(G_copy, initial_position, sigma=sigma)
    perturbed_G = calculate_edge_attributes(G_copy, perturbed_position)
    entropy_perturbed = ox.bearing.orientation_entropy(perturbed_G)
    return [sigma, rep, entropy_perturbed]


def simulate_entropy(G, initial_position, sigmas, n_repeats, output_path, rerun_simulation=True):
    """
    Simulate orientation entropy over multiple perturbation levels (sigma values).
    Optionally loads from file if rerun_simulation=False and file exists.
    """
    if os.path.exists(output_path) and not rerun_simulation:
        print("Loading existing simulation results...")
        return pd.read_excel(output_path)

    print("Running entropy simulations...")
    tasks = [(sigma, rep, G, initial_position) for sigma in sigmas for rep in range(n_repeats)]
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for result in tqdm(executor.map(run_simulation, tasks), total=len(tasks)):
            results.append(result)

    df = pd.DataFrame(results, columns=["Sigma", "Repetition", "Entropy"])
    df.to_excel(output_path, index=False)
    return df


# ------------------------------
# STEP 4: RESULTS VISUALIZATION
# ------------------------------

def plot_entropy_boxplot(df, sigmas, output_path):
    """
    Create and save a boxplot of entropy values per sigma.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Sigma", y="Entropy", data=df, color='lightblue')
    plt.title("Orientation Entropy vs. Perturbation Level (σ)")
    plt.xlabel("Sigma (σ)")
    plt.ylabel("Orientation Entropy")
    xticks = np.arange(0, len(sigmas), step=10)
    plt.xticks(xticks, [f"{sigmas[i]:.3f}" for i in xticks], rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_entropy_mean_std(df, output_path):
    """
    Create and save a line plot of mean entropy per sigma with ±1 standard deviation as a shaded band.
    """
    summary = df.groupby("Sigma")["Entropy"].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(summary["Sigma"], summary["mean"], label="Mean Entropy", color="blue")
    plt.fill_between(
        summary["Sigma"],
        summary["mean"] - summary["std"],
        summary["mean"] + summary["std"],
        alpha=0.3,
        color="blue",
        label="±1 std dev"
    )
    plt.xlabel("Sigma (σ)")
    plt.ylabel("Orientation Entropy")
    plt.title("Mean Entropy vs. Perturbation Level (σ)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_entropy_boxplot_grouped(df, output_path, bin_size=0.01):
    """
    Create and save grouped boxplots of entropy values by binning sigma values.
    """
    df_grouped = df.copy()
    df_grouped["Sigma_bin"] = (df_grouped["Sigma"] // bin_size * bin_size).round(3)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Sigma_bin", y="Entropy", data=df_grouped, color='skyblue')
    plt.title(f"Entropy Boxplot Grouped by Sigma (bin size = {bin_size})")
    plt.xlabel("Sigma (σ) Binned")
    plt.ylabel("Orientation Entropy")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_entropy_violin_grouped(df, output_path, bin_size=0.01):
    """
    Create and save violin plots of entropy values grouped by binned sigma values.
    """
    df_grouped = df.copy()
    df_grouped["Sigma_bin"] = (df_grouped["Sigma"] // bin_size * bin_size).round(3)

    plt.figure(figsize=(12, 6))
    sns.violinplot(x="Sigma_bin", y="Entropy", data=df_grouped, inner="quartile", color="lightgreen")
    plt.title(f"Entropy Violin Plot Grouped by Sigma (bin size = {bin_size})")
    plt.xlabel("Sigma (σ) Binned")
    plt.ylabel("Orientation Entropy")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ------------------------------
# STEP 5 (OPTIONAL): GRAPH EXPORT
# ------------------------------

def generate_representative_graphs(
    sigmas_to_plot,
    repetition_to_use,
    output_dir="simulation-project/output/representative_graphs"
):
    """
    Generate and save the graph and polar orientation histogram for selected sigma values
    at a given repetition index, without rerunning the full simulation.
    """
    # Setup base graph and positions
    G_base = create_base_graph()
    initial_position = center_node_positions(G_base)
    G_base = assign_node_attributes(G_base, initial_position)

    os.makedirs(output_dir, exist_ok=True)

    for sigma in sigmas_to_plot:
        G = deepcopy(G_base)
        perturbed_pos = perturb_graph(G, initial_position, sigma=sigma)
        G = calculate_edge_attributes(G, perturbed_pos)

        sigma_folder = os.path.join(output_dir, f"sigma_{sigma:.3f}")
        os.makedirs(sigma_folder, exist_ok=True)

        plot_graph(G, perturbed_pos, f"Perturbed Grid (σ={sigma:.3f})", os.path.join(sigma_folder, "graph.png"))

        fig, ax = ox.plot_orientation(G)
        fig.savefig(os.path.join(sigma_folder, "polar_orientation.png"))
        plt.close(fig)

    print(f"Representative graphs saved to: {output_dir}")    



# -----------------
# MAIN ENTRY POINT
# -----------------

def main(rerun_simulation=True):
    """
    Main workflow to run the simulation or load existing results,
    and generate the final analysis plot.
    """
    output_dir = "simulation-project/output"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Base graph setup
    G = create_base_graph()
    initial_position = center_node_positions(G)
    G = assign_node_attributes(G, initial_position)
    G = calculate_edge_attributes(G, initial_position)

    # Step 2: Save initial orientation and layout
    ox.bearing.orientation_entropy(G)
    fig, ax = ox.plot_orientation(G)
    fig.savefig(f"{output_dir}/initial_orientation.png")
    plt.close(fig)
    plot_graph(G, initial_position, "Initial Grid Graph", f"{output_dir}/initial_grid.png")

    # Step 3: Define experiment parameters and run simulation
    n_repeats = 30
    sigmas = np.round(np.arange(0.017, 0.5, 0.001), 3)
    results_path = f"{output_dir}/entropy_results_multiple.xlsx"

    df = simulate_entropy(G, initial_position, sigmas, n_repeats, results_path, rerun_simulation)

    # Step 4: Analyze and visualize results
    plot_entropy_boxplot(df, sigmas, f"{output_dir}/entropy_boxplot.png")
    plot_entropy_mean_std(df, f"{output_dir}/entropy_lineplot.png")
    plot_entropy_boxplot_grouped(df, f"{output_dir}/entropy_boxplot_grouped.png")
    plot_entropy_violin_grouped(df, f"{output_dir}/entropy_violinplot_grouped.png")

    # Step 6 optional: Generate representative graphs for specific sigmas
    representative_sigmas = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    generate_representative_graphs(
        sigmas_to_plot=representative_sigmas,
        repetition_to_use=0,
        output_dir=os.path.join(output_dir, "representative_graphs")
    )

    print("Analysis complete. Results saved.")

if __name__ == "__main__":
    main(rerun_simulation=False)

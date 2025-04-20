import os
import osmnx as ox
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import truncnorm
from shapely.geometry import LineString
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from copy import deepcopy



def perturb_graph(G, pos, sigma):
    """Perturb node positions using truncated normal distribution."""
    a, b = -3, 3  # truncate to ±3σ
    return {
        node: (
            x + truncnorm.rvs(a, b, loc=0, scale=sigma),
            y + truncnorm.rvs(a, b, loc=0, scale=sigma)
        )
        for node, (x, y) in pos.items()
    }

def calculate_edge_attributes(G, pos):
    """Assign length, geometry, and bearing attributes to graph edges."""
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

def plot_graph(G, pos, title, filename):
    """Plot and save the graph visualization."""
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

def run_simulation(args):
    """Función auxiliar para paralelizar el cálculo de entropía."""
    sigma, rep, G, initial_position = args
    G_copy = deepcopy(G)
    perturbed_position = perturb_graph(G_copy, initial_position, sigma=sigma)
    perturbed_G = calculate_edge_attributes(G_copy, perturbed_position)
    entropy_perturbed = ox.bearing.orientation_entropy(perturbed_G)
    return [sigma, rep, entropy_perturbed]

def main():
    # Define output directory for plots
    output_dir = "simulation-project/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a grid graph
    n_rows, n_cols = 10, 10
    G = nx.grid_2d_graph(n_rows, n_cols)
    G = nx.MultiGraph(G)
    G.graph["crs"] = "EPSG:4326"

    # Center the node positions
    center_x, center_y = (n_cols - 1) / 2, -(n_rows - 1) / 2
    initial_position = {node: (float(node[1]) - center_x, float(-node[0]) - center_y) for node in G.nodes()}

    # Assign attributes
    nx.set_node_attributes(G, {node: x for node, (x, _) in initial_position.items()}, "x")
    nx.set_node_attributes(G, {node: y for node, (_, y) in initial_position.items()}, "y")

    initial_G = calculate_edge_attributes(G, initial_position)
    ox.bearing.orientation_entropy(initial_G)

    # Save initial orientation plot
    fig, ax = ox.plot_orientation(initial_G)
    fig.savefig(f"{output_dir}/initial_orientation.png")
    plt.close(fig)
    plot_graph(initial_G, initial_position, "Initial Grid Graph", f"{output_dir}/initial_grid.png")

    n_repeats = 30

    # Define sigma ranges (con paso fino)
    sigmas = np.arange(0.017, 0.5, 0.001)
    sigmas = np.round(sigmas, 3)

    # Preparar lista de tareas (cada combinación de sigma y repetición)
    tasks = [(sigma, rep, G, initial_position) for sigma in sigmas for rep in range(n_repeats)]

    # Ejecutar en paralelo
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for result in tqdm(executor.map(run_simulation, tasks), total=len(tasks)):
            results.append(result)

    # Guardar resultados
    df = pd.DataFrame(results, columns=["Sigma", "Repetition", "Entropy"])
    df.to_excel(f"{output_dir}/entropy_results_multiple.xlsx", index=False)
    print("Simulation completed. Results saved in entropy_results_multiple.xlsx")

    # Graficar boxplot (simplificado en eje x)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Sigma", y="Entropy", data=df, color='lightblue')
    plt.title("Variación de la entropía de orientación por nivel de perturbación (σ)")
    plt.xlabel("Sigma (σ)")
    plt.ylabel("Entropía de orientación")
    xticks = np.arange(0, len(sigmas), step=10)
    plt.xticks(xticks, [f"{sigmas[i]:.3f}" for i in xticks], rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/entropy_boxplot.png")
    plt.close()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
import os
import osmnx as ox
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString

def perturb_graph(G, pos, epsilon=0.1):
    """Perturb node positions randomly within a controlled range."""
    return {node: (x + np.random.uniform(-epsilon, epsilon), 
                   y + np.random.uniform(-epsilon, epsilon)) 
            for node, (x, y) in pos.items()}

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

def main():

    # Define output directory for plots
    oputput_dir = "simulation-project/output"
    if not os.path.exists(oputput_dir):
        os.makedirs(oputput_dir)

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
    entropy_initial = ox.bearing.orientation_entropy(G)
    
    # Save initial orientation plot
    fig, ax = ox.plot_orientation(initial_G)
    fig.savefig(f"{oputput_dir}/initial_orientation.png")
    plt.close(fig)
    plot_graph(initial_G, initial_position, "Initial Grid Graph", f"{oputput_dir}/initial_grid.png")
    
    # Perturb the network
    perturbed_position = perturb_graph(G, initial_position, epsilon=0.5)
    perturbed_G = calculate_edge_attributes(G, perturbed_position)
    
    entropy_perturbed = ox.bearing.orientation_entropy(perturbed_G)
    
    # Save perturbed orientation plot
    fig, ax = ox.plot_orientation(initial_G)
    fig.savefig(f"{oputput_dir}/perturbed_orientation.png")
    plt.close(fig)
    plot_graph(perturbed_G, perturbed_position, "Perturbed Grid Graph", f"{oputput_dir}/perturbed_grid.png")

if __name__ == "__main__":
    main()
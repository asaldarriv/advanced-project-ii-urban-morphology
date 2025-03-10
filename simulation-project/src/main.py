import osmnx as ox
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString

def perturb_graph(G, pos, epsilon=0.1):
    """Perturb node positions randomly within a controlled range."""
    perturbed_pos = {node: (x + np.random.uniform(-epsilon, epsilon), 
                             y + np.random.uniform(-epsilon, epsilon)) 
                     for node, (x, y) in pos.items()}
    return perturbed_pos

# Assign attributes to the edges
def calculate_edge_attributes(G, pos):
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

n_rows, n_cols = 10, 10
G = nx.grid_2d_graph(n_rows, n_cols)
G = nx.MultiGraph(G)
G.graph["crs"] = "EPSG:4326"

# Determine the offset to center the node (0,0)
center_x, center_y = (n_cols - 1) / 2, -(n_rows - 1) / 2

# Assign positions with the new center
initial_position = {node: (float(node[1]) - center_x, float(-node[0]) - center_y) for node in G.nodes()}
nx.set_node_attributes(G, {node: x for node, (x, _) in initial_position.items()}, "x")
nx.set_node_attributes(G, {node: y for node, (_, y) in initial_position.items()}, "y")

# Initial entropy calculation
initial_G = calculate_edge_attributes(G, initial_position)
entropy_initial = ox.bearing.orientation_entropy(G)
print(f"Initial Orientation Entropy: {entropy_initial:.4f}")

########### Plot the initial graph
ox.plot_orientation(initial_G)
fig, ax = plt.subplots()
for u, v, data in initial_G.edges(data=True):
    x1, y1 = initial_position[u]
    x2, y2 = initial_position[v]
    ax.plot([x1, x2], [y1, y2], color='black')

# Plot the nodes
x_vals = [x for x, y in initial_position.values()]
y_vals = [y for x, y in initial_position.values()]
ax.scatter(x_vals, y_vals, color='red')

ax.set_aspect('equal')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Perturbed Grid Graph')
plt.show()

# Perturb the network
perturbed_position = perturb_graph(G, initial_position, epsilon=0.5)
perturbed_G = calculate_edge_attributes(G, perturbed_position)

# Recalculate entropy after perturbation
entropy_perturbed = ox.bearing.orientation_entropy(perturbed_G)
print(f"Perturbed Orientation Entropy: {entropy_perturbed:.4f}")

########### Plot the perturbed graph
ox.plot_orientation(perturbed_G)
fig, ax = plt.subplots()
for u, v, data in perturbed_G.edges(data=True):
    x1, y1 = perturbed_position[u]
    x2, y2 = perturbed_position[v]
    ax.plot([x1, x2], [y1, y2], color='black')

# Plot the nodes
x_vals = [x for x, y in perturbed_position.values()]
y_vals = [y for x, y in perturbed_position.values()]
ax.scatter(x_vals, y_vals, color='red')

ax.set_aspect('equal')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Perturbed Grid Graph')
plt.show()

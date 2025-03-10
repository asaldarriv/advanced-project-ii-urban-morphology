import osmnx as ox
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString

n_rows, n_cols = 10, 10
G = nx.grid_2d_graph(n_rows, n_cols)
G = nx.MultiGraph(G)
G.graph["crs"] = "EPSG:4326"

# Determine the offset to center the node (0,0)
center_x, center_y = (n_cols - 1) / 2, -(n_rows - 1) / 2

# Assign positions with the new center
pos = {node: (float(node[1]) - center_x, float(-node[0]) - center_y) for node in G.nodes()}
nx.set_node_attributes(G, {node: x for node, (x, _) in pos.items()}, "x")
nx.set_node_attributes(G, {node: y for node, (_, y) in pos.items()}, "y")

# Assign attributes to the edges
for u, v in G.edges():
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    
    length = np.hypot(x2 - x1, y2 - y1)
    geom = LineString([(x1, y1), (x2, y2)])
    bearing = ox.bearing.calculate_bearing(y1, x1, y2, x2)
    
    G[u][v][0]["length"] = float(length)
    G[u][v][0]["geometry"] = geom
    G[u][v][0]["bearing"] = float(bearing)

# Calculate orientation entropy
entropy = ox.bearing.orientation_entropy(G)
print(f"Orientation entropy: {entropy:.4f}")

# Plot the graph: TO-DO: Not working using ox.plot.plot_graph
# fig, ax = ox.plot.plot_graph(G)

# Plot the graph with matplotlib
fig, ax = plt.subplots()
for u, v, data in G.edges(data=True):
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    ax.plot([x1, x2], [y1, y2], color='black')

# Plot the nodes
x_vals = [x for x, y in pos.values()]
y_vals = [y for x, y in pos.values()]
ax.scatter(x_vals, y_vals, color='red')

ax.set_aspect('equal')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Grid Graph')
plt.show()
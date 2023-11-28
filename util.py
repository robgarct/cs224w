#the util file

from graph import CVPRGraph, BaseGraph, SolverGraph
import numpy as np
import pickle5 as pickle

def generate_graph_scratch(num_nodes=10, min_cap=1, max_cap=10):
    """
    generates graph from scratch
    Args:
        num_nodes: (int) the number of nodes in the graph excluding the depot
        min_cap: (int) the minimum capacity
        max_cap: (int) the maximum capacity
    Returns:
        (Graph) graph
    """

    g = CVPRGraph(num_nodes)

    # add graph location
    all_loc = np.random.uniform(size=(num_nodes, 2))
    location_dict_x = {node: all_loc[node][0] for node in g.get_nodes()}
    location_dict_y = {node: all_loc[node][1] for node in g.get_nodes()}
    g.assign_location(location_dict_x, location_dict_y)

    # add graph capacity
    capacity = np.random.randint(min_cap, max_cap, size=num_nodes)
    capacity_dict = {g.depot_node: 0}
    for node in g.get_nodes():
        if node != g.depot_node:
            capacity_dict[node] = capacity[node]
    g.set_capacity(capacity_dict)

    # add graph cost
    distance_dict = {}
    for e in g.get_edges():
        node1, node2 = g.get_nodes()[e[0]], g.get_nodes()[e[1]]
        loc1 = np.array([node1["x"], node1["y"]])
        loc2 = np.array([node2["x"], node1["y"]])
        dist = np.linalg.norm(loc1 - loc2)
        distance_dict[e] = {"cost": dist}
    g.add_cost(distance_dict)

    return g

def generate_multiple_graphs(N, num_nodes):
    """
    Args:
        N: (int) the number of graphs to generate
        num_nodes: (int) number of nodes in the graph
    """

    return [generate_graph_scratch(num_nodes) for i in range(N)]

def create_travel_graph(visited_nodes, loc_array=None):
    """
    create a graph for the visited nodes
    Args:
        visited_nodes: (List[int]) list of visited nodes
        loc_array: (np.ndarray) the location of the nodes. shape: (num nodes, 2)
                                array order needs to match the node order
    Returns:
        (SolverGraph)
    """

    g = SolverGraph()
    u = g.depot_node
    for v in visited_nodes:
        g.add_edge(u, v)
        u = v
    if loc_array is not None:
        # checker
        loc_x = {}
        loc_y = {}
        for n in range(loc_array.shape[0]):
            loc_x[n], loc_y[n] = loc_array[n][0], loc_array[n][0]
        g.assign_location(loc_x, loc_y)
    g.draw_graph()





#the util file

from graph import CVRPGraph, GraphCollection
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

    g = CVRPGraph(num_nodes)

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

def create_graph_solver_solution(graph_details, solution_details):
    """
    create a graph from the solver solution output
    Args:
        graph_details: (List[List]) the details of the input graph
        solution_details: (List) the solution details
    Returns:
        GraphCollection
    """
    depot_loc = graph_details[0]
    all_loc = {0: depot_loc}
    all_capacity = {0: 0}
    for i in range(len(graph_details[1])):
        all_loc[i+1] = graph_details[1][i]
        all_capacity[i+1] = graph_details[2][i]
    order_nodes = solution_details[1]

    col = GraphCollection(depot_loc[0], depot_loc[1])
    col.add_multiple_nodes(order_nodes, all_loc, all_capacity)

    return col


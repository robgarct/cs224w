## Classes for generating CVRP data

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.utils.convert import from_networkx
from typing import List
from torch_geometric.data import Data

import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class Solution:
    """Wrapper of Graph: "Generator" and solution to it"""
    # TODO: Code this 
    pass

class CVRPGraph:
    """Wrapper around graph, this is what the Generator returns
    """
    def export_pyg(self) -> Data:
        # not sure about this, but I think the "y" values for each partial
        # solution graph should be the next node to be visited.
        pass

class SolutionInstance:
    # TODO: implement this
    def __init__():
        graph: CVRPGraph = None
        solution: List = []

    def get_partial_solutions() -> List[CVRPGraph]:
        # TODO: this is not a good idea, figure out a nice way of generating
        # Partial CVRPGraph solutions without the need of k
        # return a graph with the edges of the solution up to the kth step
        pass


class Generator:
    """
    Base class for generating CVRP data

    TODO: This class is having 2 functionalities rn which are Generator and Graph.
    Let's split that to Generator only and CVRPGraph only.
    """
    def __init__(self, n):
        """
        Args:
             n: (int) number of nodes excluding the depot noe
        """
        self.n = n + 1
        self.depot_node = 0 #by default
        self.G = nx.complete_graph(n+1)
        self.assign_location()
        self.add_cost()
        self.add_capacity()
        self.add_depot_distance()

    def assign_location(self):
        """
        assigns the location in Euclidean plane for nodes
        """
        locations = np.round(np.random.uniform(size=(self.n, 2)), 4)

        all_loc = np.random.uniform(size=(self.n, 2))
        location_dict = {node:all_loc[node] for node in self.G.nodes()}
        nx.set_node_attributes(self.G, location_dict, name="location")

        location_dict_x = {node:all_loc[node][0] for node in self.G.nodes()}
        location_dict_y = {node: all_loc[node][1] for node in self.G.nodes()}
        nx.set_node_attributes(self.G, location_dict_x, name="x")
        nx.set_node_attributes(self.G, location_dict_y, name="y")

    def add_cost(self):
        """
        adds edge weight, which corresponds to the distance between
        the connecting nodes
        """

        distance_dict = {}
        for e in self.G.edges():
            node1, node2 = self.G.nodes()[e[0]], self.G.nodes()[e[1]]
            loc1 = np.array([node1["x"], node1["y"]])
            loc2 = np.array([node2["x"], node1["y"]])
            dist = np.linalg.norm(loc1 - loc2)
            distance_dict[e] = {"cost":dist}
        nx.set_edge_attributes(self.G, distance_dict)

    def add_capacity(self, min_cap=1, max_cap=10):
        """
        adds product capacity each node
        Args:
            min_cap: (int) minimum capacity
            max_cap: (int) maximum capacity
        """

        capacity = np.random.randint(min_cap, max_cap, size=self.n)

        capacity_dict = {self.depot_node:0}
        for node in self.G.nodes():
            if node != self.depot_node:
                capacity_dict[node] = capacity[node]
        nx.set_node_attributes(self.G, capacity_dict, name="capacity")

    def draw_graph(self, folder="figure", name="sample_graph"):
        """
        draws the graph
        """

        f = nx.draw(self.G, with_labels=True)
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        if ".pdf" not in name:
            name = name +".pdf"
        full_path = folder_path / name
        plt.savefig(full_path)
        print("saved")

    def add_depot_distance(self):
        """
        adds depot distance attribute
        """

        depot_dist_dict = {self.depot_node:0}
        for node in self.G.nodes():
            if node != self.depot_node:
                depot_dist = self.G.edges()[self.depot_node, node]["cost"]
                depot_dist_dict[node] = depot_dist
        nx.set_node_attributes(self.G, depot_dist_dict, name="depot_dist")

    def export_pyg(self):
        """
        Returns:
            (torch_geometric.data.Data)
        """
        return from_networkx(self.G)

    def get_graph(self):
        """
        Returns:
             nx.graph.Graph
        """

        return self.G

    def export_cytoscape(self, folder="data", output_filename="g_cytoscape"):
        """
        export data in cytoscape format
        """

        data_json = nx.cytoscape_data(self.G)
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)

        if '.json' not in output_filename:
            output_filename += '.json'
        full_path = folder_path / output_filename
        with open(full_path, 'w') as f:
            json.dump(data_json, f, cls=NpEncoder)
        print("cyjs file exported to: {}".format(output_filename))

    def get_location(self, node_id):
        """
        get location of a given node
        Args:
             node_id : (int / List[int]) node_ide
        Returns:
            np.ndarray
        """

        array = None
        if isinstance(node_id, int):
            array = np.zeros(2, )
            x, y = self.G.nodes()[node_id]["x"], self.G.nodes()[node_id]["y"]
            array[0], array[1] = x, y
        else:
            array = np.zeros((len(node_id), 2))
            for i in range(len(node_id)):
                x, y = self.G.nodes()[node_id[i]]["x"], self.G.nodes()[node_id[i]]["y"]
                array[i][0] = x
                array[i][1] = y
        return array

    def get_capacities(self):
        """
        export the product capacity of all nodes
        Return:
            (np.ndarray)
        """

        capacity = []
        for node in self.G.nodes():
            if node != self.depot_node:
                capacity.append(self.G.nodes()[node]["capacity"])

        return np.array(capacity)

    def export_solver_data(self):
        """
        Exports data needed by the solver
        Returns:
             (np.ndarray, np.ndarray, np.ndarray) depo location, location of other nodes,
                                                   capacity array
        """
        depo = self.get_location(self.depot_node)
        non_depo_nodes = []
        for node in self.G.nodes():
            if node != self.depot_node:
                non_depo_nodes.append(node)
        graph = self.get_location(non_depo_nodes)
        demand = self.get_capacities()

        return depo, graph, demand


#depo: location(x, y)
#graphs: n x 2 array: excluding depot loc
#demand n x 1 array with of capacity




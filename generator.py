## Classes for generating CVRP data

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.utils.convert import from_networkx

class Generator:
    """
    Base class for generating CVRP data
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

    def add_cost(self):
        """
        adds edge weight, which corresponds to the distance between
        the connecting nodes
        """

        distance_dict = {}
        for e in self.G.edges():
            node1, node2 = self.G.nodes()[e[0]], self.G.nodes()[e[1]]
            dist = np.linalg.norm(node1["location"]- node2["location"])
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
        capacity_dict = {node: capacity[node] for node in self.G.nodes()}
        nx.set_node_attributes(self.G, capacity_dict, name="capacity")

        #for node in self.G.nodes():
        #    print(node)
        #    print(self.G.nodes()[node]["capacity"])

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





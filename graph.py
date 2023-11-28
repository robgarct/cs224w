## Classes for generating CVRP data

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.utils.convert import from_networkx
import pickle5 as pickle

'''
class Solution:
    """Wrapper of Graph: "Generator" and solution to it"""
    # TODO: Code this
    pass

class CVRPGraph:
    """Wrapper around graph, this is what the Generator returns
    """
    pass

class SolutionInstance:
    # TODO: implement this
    def __init__():
        graph: CVRPGraph = None
        solution: List = []

    def get_kth_step(k: int) -> CVRPGraph:
        # TODO: this is not a good idea, figure out a nice way of generating
        # Partial CVRPGraph solutions without the need of k
        # return a graph with the edges of the solution up to the kth step
        pass
'''

class BaseGraph:
    """
    Base class for CVPR graph
    Attributes:
        depot_node: (int) the depot node id
        G: (nx.Graph) the graph object
    """
    def __init__(self):

        self.depot_node = 0
        self.G = nx.Graph()
        self.G.add_node(self.depot_node)

    def assign_location(self, loc_x_dict, loc_y_dict):
        """
        assigns the location in Euclidean plane for nodes
        Args:
            loc_x_dict: (dict) (int) node id -> (float) x-coordinates
            loc_y_dict: (dict) (int) node id -> (float) y-coordinates
        """

        nx.set_node_attributes(self.G, loc_x_dict, name="x")
        nx.set_node_attributes(self.G, loc_y_dict, name="y")

    def add_cost(self, cost_dict):
        """
        adds the cost, which corresponds to the distance between
        the connecting nodes
        """

        nx.set_edge_attributes(self.G, cost_dict)
    def set_capacity(self, capacity_dict):
        """
        adds product capacity
        Args:
            capacity_dict: (dict) (int) node id -> (int) capacity
        """

        nx.set_node_attributes(self.G, capacity_dict, name="capacity")

    def add_depot_distance(self):
        """
        computes depot distance attribute
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

    def get_location(self, node_id):
        """
        get location of a given node
        Args:
             node_id : (int) node_id
        Returns:
            np.ndarray
        """
        loc  = np.asarray([self.G.nodes()[node_id]["x"],
                           self.G.nodes()[node_id]["y"]])

        return loc

    def get_nodes(self):
        """
        returns the nodes of the graph
        """

        return self.G.nodes()

    def get_edges(self):
        """
        returns the edges of the graph
        """

        return self.G.edges()
    def get_capacity(self, node_id):
        """
        export the product capacity of all nodes
        Return:
            float
        """

        return self.G.nodes()[node_id]["capacity"]


    def draw_graph(self, pos=None, folder="figure", name="sample_graph"):
        """
        draws the graph
        """
        if pos is None:
            f = nx.draw(self.G, with_labels=True)
        else:
            nx.draw_networkx(self.G, pos)
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        if ".pdf" not in name:
            name = name +".pdf"
        full_path = folder_path / name
        plt.savefig(full_path)
        print("saved")

    def get_num_nodes(self):
        """
        Returns:
            (int) the number of nodes in the graph
        """

        return self.G.number_of_nodes()


class CVPRGraph(BaseGraph):
    """
    Base class for CVPR graph object
    """

    def __init__(self, n):
        """
        Args:
            n: (int) number of nodes in the graph
        """
        super().__init__()
        self.n = n
        self.G = nx.complete_graph(n)

    def export_solver_format(self):
        """
        Exports data needed by the solver
        Returns:
             (np.ndarray, np.ndarray, np.ndarray) depo location, location
                                                  of other nodes, capacity array
        """
        depo = self.get_location(self.depot_node)
        demand = np.zeros(self.n-1)
        graph = np.zeros((self.n-1, 2))
        non_depo_nodes = [node for node in self.G.nodes() if node != self.depot_node]
        for i in range(len(non_depo_nodes)):
            node = non_depo_nodes[i]

            graph[i] = self.get_location(node)
            demand[i] = self.get_capacity(node)

        return depo, graph, demand

class SolverGraph(BaseGraph):
    """
    Base class for the graph created from the solver solution
    """

    def __init__(self):
        """
        Args:
            n: (int) number of nodes in the graph
        """
        super().__init__()

    def add_edge(self, u, v):
        """
        adds an edge to the graph
        Args:
            u: (int) node id
            v: (int) node id
        """

        self.G.add_edge(u, v)

    def add_node(self, u, attributes=None):
        """
        adds a node to the graph
        Args:
            u: (int) node id
            attributes: (dict)
        """

        self.G.add_node(u, attributes)


#if __name__ =="__main__":

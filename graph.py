## Classes for generating CVRP data

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.utils.convert import from_networkx
import pickle5 as pickle
from copy import deepcopy

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
        self.graph_label = None

    def assign_location(self, loc_x_dict, loc_y_dict):
        """
        assigns the location in Euclidean plane for nodes
        Args:
            loc_x_dict: (dict) (int) node id -> (float) x-coordinates
            loc_y_dict: (dict) (int) node id -> (float) y-coordinates
        """

        nx.set_node_attributes(self.G, loc_x_dict, name="x")
        nx.set_node_attributes(self.G, loc_y_dict, name="y")

    def add_node(self, node_id, x=None, y=None, c=None):
        """
        adds a single node
        Args:
            node_id: (int) the id of the node to add
            x: (float) x-co-ordinate of the node
            y: (float) y-co-ordinate of the node
            c: (int) the product capacity of the node
        """

        self.G.add_node(node_id)
        if x is not None and y is not None:
            self.assign_location({node_id:x}, {node_id:y})

        if c is not None:
            self.set_capacity({node_id:x})

    def add_edge(self, node1, node2):
        """
        adds an edge between the given nodes
        Args:
            node1: (int)
            node2: (int)
        """

        self.G.add_edge(node1, node2)

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

    ## TODO: also assign label
    def export_pyg(self):
        """
        Returns:
            (torch_geometric.data.Data)
        """

        pyg_graph =  from_networkx(self.G)

        ### this might work
        #pyg_graph["label"] = self.graph_label 
        return pyg_graph

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

    def set_graph_label(self, lab):
        """
        set label for the graph
        Args:
            lab:
        """
        self.graph_label = lab
    def get_graph_label(self):
        """
        Returns:
             the graph label
        """

        return self.graph_label

class CVRPGraph(BaseGraph):
    """
    Base class for CVPR graph object.
    Creates a fully connected graph with nodes representing locations.
    The graph also serves as the training data
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

class GraphCollection:
    """
    Base class for the storing graphs
    Has functionalities to build the graph sequentially
    """

    def __init__(self, depot_loc_x, depot_loc_y, depot_capacity=0):
        """
        depot_loc_x: (float) x-coordinate of depot_location
        depot_loc_y: (float) y-coordinate of depot_location
        """
        self.start = BaseGraph()
        self.start.assign_location({0:depot_loc_x}, {0:depot_loc_y})
        self.start.set_capacity({0: depot_capacity})
        self.visited_nodes = [self.start.depot_node]
        self.all_graphs = []
        self.curr_graph = self.start

    ## TODO: Graph label
    def add_node(self, node_id, loc_x=None, loc_y=None, capacity=None):
        """
        adds a single node and connects it to the previously visited node
        Args:
            node_id: (int) the id of the node to add
            loc_x: (float) x-co-ordinate of the node
            loc_y: (float) y-co-ordinate of the node
            capacity: (int) the product capacity of the node
        """

        self.all_graphs.append(self.curr_graph)
        self.curr_graph = deepcopy(self.curr_graph)
        self.curr_graph.add_node(node_id, loc_x, loc_y, capacity)
        self.curr_graph.add_edge(self.visited_nodes[-1], node_id)
        self.visited_nodes.append(node_id)

    def get_current_graph(self):
        """
        Returns: (BaseGraph)
        """

        return self.curr_graph

    def get_kth_graph(self, k):
        """
        gets the graph corresponding to kth step
        Args:
            k: (float)

        Returns: (BaseGraph)
        """
        if k >= len(self.all_graphs):
            raise ValueError("{} must be less than {}".format(
                k, len(self.all_graphs)))
        else:
            return self.all_graphs[k]

    def add_multiple_nodes(self, node_ids, loc_dict=None, capacity_dict=None):
        """
        add multiple_nodes to the graph
        Args:
            node_ids: (List[int]) the nodes to add
            loc_dict: (dict) (int) node_id > List[float, float] x_loc, y_loc
            capacity_dict: (dict) (int) node_id > (int) capacity
        """

        for i in range(len(node_ids)):
            n = node_ids[i]
            if n in loc_dict and n in capacity_dict:
                self.add_node(n, loc_dict[n][0], loc_dict[n][0], capacity_dict[n])
            elif n in loc_dict:
                self.add_node(n, loc_dict[n][0], loc_dict[n][0])
            elif n in capacity_dict:
                self.add_node(n, capacity=capacity_dict[n])
            else:
                self.add_node(n)

    def get_all_graphs(self):
        """
        Returns:
            (List[BaseGraph])
        """

        return self.all_graphs

    def __str__(self):

        return "Graph Collection consisting of {} graphs. nodes: {}".format(len(self.all_graphs),
                                  self.visited_nodes)

## Classes for generating CVRP data
import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data
# import pickle5 as pickle
from copy import deepcopy
import torch
import math

import pandas as pd
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
CAPACITIES = {
        20: 30,
        50: 40,
        100: 50
}
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
            self.set_capacity({node_id:c})

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

        nx.set_edge_attributes(self.G, cost_dict, name="cost")

    def add_edge_distance(self):
        """
        add distance between edges
        """
        dist_dict = {}
        for u, v in self.G.edges():
            loc_u, loc_v = self.get_location(u), self.get_location(v)
            dist_uv = np.sum(np.square(loc_u - loc_v)) ** 0.5
            dist_dict[(u, v)] = dist_uv

        self.add_cost(dist_dict)

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
                loc_depot, loc_v = self.get_location(self.depot_node), self.get_location(node)
                depot_dist = np.sum(np.square(loc_depot - loc_v)) ** 0.5
                depot_dist_dict[node] = round(depot_dist, 3)
        nx.set_node_attributes(self.G, depot_dist_dict, name="depot_dist")

    ## TODO: also assign label
    def export_pyg(self):
        """
        Returns:
            (torch_geometric.data.Data)
        """
        #pyg_graph =  from_networkx(self.G)
        edge_list = []
        edge_attributes = []
        i = 0
        for edge in self.G.edges():
            edge_list.append(edge)
            i += 1
        edge_list = torch.LongTensor(edge_list).T
        #edge_attributes = torch.tensor(edge_attributes)

        n = self.G.number_of_nodes()
        feature_matrix = np.zeros((n, 4))
        loc_x, loc_y, cap, dist_depot = np.zeros(n, ), np.zeros(n, ), np.zeros(n, ), np.zeros(n, )
        n = 0
        for node in self.G.nodes():        
            loc_x[n] = nx.get_node_attributes(self.G, "x")[node]
            loc_y[n] = nx.get_node_attributes(self.G, "y")[node]
            cap[n] = nx.get_node_attributes(self.G, "capacity")[node]
            if n==0: 
                depo_x = nx.get_node_attributes(self.G, "x")[node]
                depo_y = nx.get_node_attributes(self.G, "y")[node]
                dist_depot[n] = 0
            else:
                dist_depot[n] = math.sqrt((depo_x-loc_x[n])**2 + (depo_y-loc_y[n])**2)
            n += 1
        feature_matrix[:, 0], feature_matrix[:, 1] = loc_x, loc_y
        feature_matrix[:, 2] = cap
        feature_matrix[:, 3] = dist_depot
        feature_matrix = torch.FloatTensor(feature_matrix)

        graph_label, prev_node, vehicle_cap = self.get_graph_parameters()
        pyg_graph = Data(x=feature_matrix,
                         edge_index=edge_list,
                         prev_node=torch.tensor(prev_node),
                         vehicle_cap=torch.tensor(vehicle_cap),
                         edge_attr={})
        pyg_graph["y"] = graph_label
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


    def draw_graph(self, with_pos=True, pos=None, folder="figure", name="sample_graph",
                   axis=None, labels=False):
        """
        draws the graph
        """
        if with_pos:

            pos = {node: self.get_location(node) for node in self.G.nodes()}
            f = nx.draw_networkx(self.G, pos=pos, with_labels=False, node_size=20, font_size=5,
                                 ax=axis)
            if labels:
                nx.draw_networkx_labels(self.G, pos, font_size=8, verticalalignment="bottom",
                                        ax=axis)
        else:
            if pos is None:
                pos = nx.spring_layout(self.G, k=0.3 * 1 / np.sqrt(len(self.G.nodes())),
                                       iterations=20)
                nx.draw_networkx(self.G, pos, node_size=100, font_size=8)
                #edge_labels = nx.get_edge_attributes(self.G, 'cost')
                #nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels,
                 #                            label_pos=0, font_size=5)
            else:
                nx.draw_networkx(self.G, pos)
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        if ".png" not in name:
            name = name +".png"
        full_path = folder_path / name
        plt.savefig(full_path)
        print("saved")

    def get_num_nodes(self):
        """
        Returns:
            (int) the number of nodes in the graph
        """

        return self.G.number_of_nodes()

    def set_graph_parameters(self, lab, prev, v_cap):
        """
        set label for the graph
        Args:
            lab:
        """
        self.graph_label = lab
        self.prev = prev
        self.vehicle_cap = v_cap
        
    def get_graph_parameters(self):
        """
        Returns:
             the graph label
        """

        return self.graph_label, self.prev, self.vehicle_cap

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

    def __init__(self, loc_dict_x, loc_dict_y, capacity_dict):
        """
        depot_loc_x: (float) x-coordinate of depot_location
        depot_loc_y: (float) y-coordinate of depot_location
        """
        
        self.start = BaseGraph()
        for key in loc_dict_x:
            self.start.add_node(key)
        self.start.assign_location(loc_dict_x, loc_dict_y)
        self.start.set_capacity(capacity_dict)
        #print("nodes", self.start.G.nodes())
        self.visited_nodes = [self.start.depot_node]
        self.curr_graph = self.start
        self.all_graphs = []
        depot_cap = CAPACITIES[self.num_nodes-1]
        self.vehicle_capacity_map = [depot_cap]

    @property
    def num_nodes(self):
        return nx.number_of_nodes(self.curr_graph.G)
        
    ## TODO: Graph label
    def update_node(self, node_id, loc_x=None, loc_y=None, capacity=None):
        """
        adds a single node and connects it to the previously visited node
        Args:
            node_id: (int) the id of the node to add
            loc_x: (float) x-co-ordinate of the node
            loc_y: (float) y-co-ordinate of the node
            capacity: (int) the product capacity of the node
        """
        
        dist = None
        prev_node = self.visited_nodes[-1]
        node_demand = self.curr_graph.get_capacity(node_id)
        v_cap = self.vehicle_capacity_map[-1]
       
        assert prev_node is not None
        self.curr_graph.set_graph_parameters(node_id, prev_node, v_cap)
        self.all_graphs.append(self.curr_graph)
        self.curr_graph = deepcopy(self.curr_graph)
        self.curr_graph.add_edge(self.visited_nodes[-1], node_id)
        self.visited_nodes.append(node_id)
        unique_nodes = np.unique(self.visited_nodes)
        ## implies all nodes have been visited. we connect the last node to the depot
        if len(unique_nodes) == len(self.curr_graph.get_nodes()):
            #print(node_id)
            #print("equal")
            prev_node = self.visited_nodes[-1]
            self.curr_graph.set_graph_parameters(self.curr_graph.depot_node, prev_node, 0)
            self.all_graphs.append(self.curr_graph)

            #self.curr_graph.add_edge(self.visited_nodes[-2], self.visited_nodes[-1])
        
        # vehicle capacity after the vehicle has passed from this node
        v_cap = self.vehicle_capacity_map[0] if node_id==0 else v_cap-node_demand
        v_cap = max(0,v_cap)
        self.vehicle_capacity_map.append(v_cap)
        ## Alternative solution

    def is_complete(self):
        return len(set(self.visited_nodes)) == self.num_nodes

    def get_current_unsolved_graph(self):
        # Returns the current unsolved graph - used for inference.
        prev_node = self.visited_nodes[-1]
        v_cap = self.vehicle_capacity_map[-1]
        curr_graph = deepcopy(self.curr_graph)
        # Since we don't have a node label during inference, we set a dummy
        # value of 1000.
        curr_graph.set_graph_parameters(10000, prev_node, v_cap)
        return curr_graph

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

    def get_k_partial_solutions(self, k):
        """
        return k partial solutions
        Args:
            k: (int)

        Returns:
            [List[BaseGraph]]
        """
        if k > len(self.all_graphs):
            print("{} is bigger than total number of available graphs. "
                  "Returning all graphs in the list")
            return self.all_graphs
        else:
            sample = random.sample(range(1, len(self.all_graphs)), k)
            return [self.all_graphs[i] for i in sample]

    def add_multiple_nodes(self, node_ids):
        """
        add multiple_nodes to the graph
        Args:
            node_ids: (List[int]) the nodes to add
        """
        for i in range(len(node_ids)):
            n = node_ids[i]
            self.update_node(n)

    def get_all_graphs(self):
        """
        Returns:
            (List[BaseGraph])
        """

        return self.all_graphs[1:]

    def get_full_solution_cost(self):
        assert self.is_complete()
        # ensure depot is at beginning and end
        sol = [0] + self.visited_nodes + [0]
        prev_n = 0
        dist = 0
        for n in sol:
            prev_n_loc = self.start.get_location(prev_n)
            n_loc = self.start.get_location(n)
            dist += np.linalg.norm(prev_n_loc - n_loc)
            prev_n = n
        return dist
        

    def __str__(self):

        return "Graph Collection consisting of {} graphs. nodes: {}".format(len(self.all_graphs),
                                  self.visited_nodes)

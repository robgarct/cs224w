"""
Model implementation
"""
import torch
from torch import nn
from torch_geometric.data.batch import Batch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class Model(nn.Module):
    """ model class
    this uses a GCN from pytorch geometric or whatever. However, it
    implements other useful functionality for training and inference.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(Model, self).__init__()
        
        ## Number of layers
        self.num_layers = num_layers

        # A list of GCNConv layers
        self.convs = torch.nn.ModuleList()
         # A list of 1D batch normalization layers
        self.bns = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers-2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # pooling layer for each graph
        self.pool = global_mean_pool
        
        # The log softmax layer
        self.softmax = torch.nn.LogSoftmax(dim=1)
        
        ## Apply dropout
        self.dropout = dropout

    @property
    def device(self):
        return next(self.parameters()).device
    
    def get_visited_nodes_and_update_embeddings(self, x: torch.Tensor, batched_graphs: Batch):
        """ Get the visited nodes and make their embeddings to be -Inf so that the softmax is 0 """
        batch_edge_index = batched_graphs.edge_index
        
        for graph_idx in range(batched_graphs.num_graphs):
            batch_mask = batched_graphs.batch == graph_idx

            # Nodes of the graph
            batch_nodes = torch.nonzero(batch_mask).flatten()
            mask = torch.isin(batch_edge_index[0], batch_nodes)
            
            # Edge index/Edges of the graph
            edge_index = batch_edge_index[:,mask]
            
            # Connected nodes are just unique nodes of the edges
            connected_nodes = torch.unique(edge_index.flatten())
            
            # No edges TODO(pulkit): Take a look at this
            if connected_nodes.shape[0] == 0:
                continue
           
            # remove the depot node, Here a heuristic is that the depot node is the node 
            # in the batch with min index
            min_value, _ = torch.min(connected_nodes, dim=0)
            connected_nodes = connected_nodes[connected_nodes!=min_value]
            
            mask = torch.isin(batch_nodes, connected_nodes)
            x[graph_idx].masked_fill_(mask,-1e10)
            
        return x
    
    def forward(self, batched_graphs : Batch):
        # receives a batched data instance:
        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Batch.html
        
        # If doing VRP_50 or any VRP_N, this should return 
        # a tensor of shape(batch_size, N) with logits. The
        # logits for nodes that are already visited should
        # be masked out with a very small number
        
        x, edge_index, batch = batched_graphs.x, batched_graphs.edge_index, batched_graphs.batch
        
        for i in range(self.num_layers-1):
          x = self.convs[i](x=x, edge_index=edge_index)
          x = self.bns[i](x)
          x = F.relu(x)
          x = F.dropout(x, p = self.dropout, training=self.training)

        x = self.convs[self.num_layers-1](x=x, edge_index=edge_index)
        
        ## pool over the batch, to get 1 embedding of size(= number of nodes) for each graph
        x = self.pool(x, batch=batch)
        ## Size of x here is (batch_size, N)
        
        ## To nullify the probs of connected nodes except the depot node
        x = self.get_visited_nodes_and_update_embeddings(x, batched_graphs)      
        return x


    def probs(self, batched_graphs : Batch):
        # use forward to return probs, something like:
        # probs = softmax(self.forward(graph))
        probs = self.softmax(self.forward(batched_graphs))
        return probs

    def next_node_probabilities():
        pass 


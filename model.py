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
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(Model, self).__init__()
        
        ## Number of layers
        self.num_layers = num_layers

        # A list of GCNConv layers
        self.convs = torch.nn.ModuleList()
         # A list of 1D batch normalization layers
        self.bns = torch.nn.ModuleList()
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers-2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        ## Apply dropout
        self.dropout = dropout

        # linear layers
        self.lin_pre = nn.Sequential(nn.Linear(input_dim, hidden_dim*2), nn.ReLU(), nn.Linear(hidden_dim*2, hidden_dim))
        self.lin_post = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(), nn.Linear(hidden_dim*2, hidden_dim))

    @property
    def device(self):
        return next(self.parameters()).device
    
    # def get_visited_nodes_and_update_embeddings(self, x: torch.Tensor, batched_graphs: Batch):
    #     """ Get the visited nodes and make their embeddings to be -Inf so that the softmax is 0 """
    #     batch_edge_index = batched_graphs.edge_index
        
    #     for graph_idx in range(batched_graphs.num_graphs):
    #         batch_mask = batched_graphs.batch == graph_idx

    #         # Nodes of the graph
    #         batch_nodes = torch.nonzero(batch_mask).flatten()
    #         mask = torch.isin(batch_edge_index[0], batch_nodes)
            
    #         # Edge index/Edges of the graph
    #         edge_index = batch_edge_index[:,mask]
            
    #         # Connected nodes are just unique nodes of the edges
    #         connected_nodes = torch.unique(edge_index.flatten())
            
    #         # No edges TODO(pulkit): Take a look at this
    #         if connected_nodes.shape[0] == 0:
    #             continue
           
    #         # remove the depot node, Here a heuristic is that the depot node is the node 
    #         # in the batch with min index
    #         min_value, _ = torch.min(connected_nodes, dim=0)
    #         connected_nodes = connected_nodes[connected_nodes!=min_value]
            
    #         mask = torch.isin(batch_nodes, connected_nodes)
    #         x[graph_idx].masked_fill_(mask,-1e10)
            
    #     return x

    def get_mask(self, batch):
        # This masking code is slow, we should make it faster
        batch_size = len(batch)
        num_nodes = batch.x.shape[0]//batch_size
        rs = torch.zeros((batch_size, num_nodes)).to(bool)
        for i in range(batch_size):
            b = batch[i]
            unique = list(set(b.edge_index[0].tolist()) | set(b.edge_index[1].tolist()))
            rs[i][torch.tensor(unique)] = True
        rs[:, 0] = False
        return rs

    def forward(self, batched_graphs : Batch):
        # receives a batched data instance:
        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Batch.html
        
        # If doing VRP_50 or any VRP_N, this should return 
        # a tensor of shape(batch_size, N) with logits. The
        # logits for nodes that are already visited should
        # be masked out with a very small number
        
        x, edge_index, batch = batched_graphs.x, batched_graphs.edge_index, batched_graphs.batch
        pre_x = x = self.lin_pre(x)

        for i in range(self.num_layers-1):
          # TODO apply skip connections?
          x = self.convs[i](x=x, edge_index=edge_index)
          x = self.bns[i](x)
          x = F.relu(x)
          if self.dropout > 0.0:
            x = F.dropout(x, p = self.dropout, training=self.training)

        x = self.convs[self.num_layers-1](x=x, edge_index=edge_index)
        x = x + pre_x
        x = F.relu(x)
        x = self.lin_post(x)
        
        batch_size = len(batched_graphs)
        idxs = torch.arange(batch_size) * 21
        prev_node = idxs+batched_graphs.prev_node

        embeds_prev = x[prev_node].reshape(batch_size, 1, -1)
        x =  x.reshape(batch_size, 21, -1)
        logits = x @ embeds_prev.transpose(-1, -2)
        mask = self.get_mask(batched_graphs)

        return logits.squeeze(-1).masked_fill(mask, -1e10)

        """
        # import pdb; pdb.set_trace()
        #size of x here is (batch_size*N, out_dim)
       
        ## Apply x.xT to get the edge embedding for all the edges
        batch_size = len(batched_graphs)
        num_nodes = (int)(x.shape[0]/batch_size)
        xxT_batch = [torch.mm(x[j*num_nodes:j*num_nodes + num_nodes,:], x[j*num_nodes:j*num_nodes + num_nodes,:].t()) for j in range(batch_size)]
        xxT_batch = torch.cat(xxT_batch,dim=0)
        #size of xxT_batch here is (batch_size*N, N)
        
        ## pool over the batch, to get 1 embedding of size(= number of nodes) for each graph
        x_pool = self.pool(xxT_batch, batch=batch)        
        ## Size of x_pool here is (batch_size, N)

        ## To nullify the probs of connected nodes except the depot node
        x_pool = self.get_visited_nodes_and_update_embeddings(x_pool, batched_graphs)
        return x_pool
        """


    def probs(self, batched_graphs : Batch):
        return  F.softmax(self.forward(batched_graphs))


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 20)
        self.lin2 = nn.Linear(20, 21)

    def get_mask(self, batch):
        batch_size = len(batch)
        rs = torch.zeros((batch_size, 21)).to(bool)
        for i in range(batch_size):
            b = batch[i]
            unique = list(set(b.edge_index[0].tolist()) | set(b.edge_index[1].tolist()))
            rs[i][torch.tensor(unique)] = True
        rs[:, 0] = False
        return rs
    
    def forward(self, batch):
        x = self.lin2(F.relu(self.lin1(batch.x)))
        batch_size = len(batch)
        idxs = torch.arange(batch_size) * 21
        prev_node = idxs+batch.prev_node
        
        embeds_prev = x[prev_node].reshape(batch_size, 1, -1)
        x =  x.reshape(batch_size, 21, -1)
        logits = x @ embeds_prev.transpose(-1, -2)
        mask = self.get_mask(batch)
        return logits.squeeze(-1).masked_fill(mask, -1e10)

    @property
    def device(self):
        return next(self.parameters()).device
    
    def probs(self, batch):
        return F.softmax(self.forward(batch), dim=-1)
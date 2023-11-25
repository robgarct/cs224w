"""
Model implementation
"""
from torch import nn
from torch_geometric.data.batch import Batch

class Model(nn.Module):
    """ model class
    this uses a GCN from pytorch geometric or whatever. However, it
    implements other useful functionality for training and inference.
    """
    pass

    def forward(batched_graphs : Batch):
        # receives a batched data instance:
        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Batch.html
        
        # If doing VRP_50 or any VRP_N, this should return 
        # a tensor of shape(batch_size, N) with logits. The
        # logits for nodes that are already visited should
        # be masked out with a very small number
        pass

    def probs(batched_graphs : Batch):
        # use forward to return probs, something like:
        # probs = softmax(self.forward(graph))
        pass

    def next_node_probabilities():
        pass 


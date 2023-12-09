from generator import *
from graph import *
from model import Model
from typing import List
from torch_geometric.data.batch import Batch

def greedy_inference_single(
    model: Model,
    graph: GraphCollection
)-> SolutionInstance:
    """Runs inference on a single graph.
    """
    graph = deepcopy(graph)
    sol = []
    while not graph.is_complete():
        curr_graph = graph.get_current_unsolved_graph()
        curr_graph = curr_graph.export_pyg()
        curr_batch = Batch.from_data_list([curr_graph])
        next_probs = model.probs(curr_batch)
        next_node = next_probs.squeeze().argmax().cpu().item()
        graph.update_node(next_node)
        sol.append(next_node)
        
    return graph

def beam_inference_single(
    model: Model,
    graph_wo_edges: CVRPGraph,
    beam_width: int = 10,
) -> SolutionInstance:
    """Runs inference on a single graph using beam search
    """
    pass


def greedy_inference_parallel(
    model: Model,
    graphs_wo_edges: List[CVRPGraph]
)-> List[SolutionInstance]:
    """Runs inference on a multiple graphs.
    """
    pass

def beam_inference_parallel(
    model: Model,
    graph_wo_edges: CVRPGraph,
    beam_width: int = 10,
) -> List[SolutionInstance]:
    """Runs inference on multiple graphs using beam search
    """
    pass
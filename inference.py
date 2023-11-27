from generator import *
from model import Model
from typing import List
from torch_geometric.data.batch import Batch

def greedy_inference_single(
    model: Model,
    graph_wo_edges: CVRPGraph
)-> SolutionInstance:
    """Runs inference on a single graph.
    """
    # start from empty sol
    # TODO(roberto): What should we do here? start from 0 or not
    curr_sol = SolutionInstance(graph_wo_edges, [0])
    while not curr_sol.sol_is_complete():
        curr_graph = curr_sol.as_partial_solution()
        curr_graph = curr_graph.export_pyg()
        curr_batch = Batch.from_data_list([curr_graph])
        next_probs = model.probs(curr_batch)
        next_node = next_probs.squeeze().argmax().cpu().item()
        curr_sol.add_node(next_node)
    # TODO(roberto): Should we also be adding the 0 node at the end?
    curr_sol.add_node(0)
    return curr_sol

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
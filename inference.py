from generator import *
from model import Model
from typing import List

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
        pass

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
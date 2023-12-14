from generator import *
from graph import *
from model import Model
from typing import List, Tuple
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


def _beam_search_recursive(model, graphs, log_probs, beam_width):
    finished_graphs = []
    all_graphs = []
    all_log_probs = []
    for graph, log_prob in zip(graphs, log_probs):
        if graph.is_complete():
            finished_graphs.append(graph)
            continue
        curr_log_probs = torch.ones(beam_width) * log_prob
        curr_graph = graph.get_current_unsolved_graph().export_pyg()
        curr_batch = Batch.from_data_list([curr_graph])
        next_probs = model.probs(curr_batch).squeeze()
        next_probs, next_nodes = torch.topk(next_probs, beam_width)
        curr_log_probs += next_probs.log()
        for next_node in next_nodes.tolist():
            _graph = deepcopy(graph)
            _graph.update_node(next_node)
            all_graphs.append(_graph)
        all_log_probs.append(curr_log_probs)
   
    print("finished", len(finished_graphs))
    print(beam_width)
    if len(finished_graphs) == beam_width:
        return finished_graphs
    
    all_log_probs = torch.cat(all_log_probs)
    top_lps, top_lps_idxs = torch.topk(all_log_probs, beam_width-len(finished_graphs))

    best_graphs = []
    best_log_probs = []
    for lp, i in zip(top_lps.tolist(), top_lps_idxs.tolist()):
        best_graphs.append(all_graphs[i])
        best_log_probs.append(lp)
    
    return finished_graphs + _beam_search_recursive(model, best_graphs, best_log_probs, beam_width - len(finished_graphs))

def beam_inference_single(
    model: Model,
    graph: GraphCollection,
    beam_width: int = 10,
) -> SolutionInstance:
    """Runs inference on a single graph using beam search
    """
    graph = deepcopy(graph)
    top_graphs = _beam_search_recursive(model, [graph], [0.0], beam_width)
    best_graph = None
    min_cost = 1e9
    for graph in top_graphs:
        cost = graph.get_full_solution_cost()
        if graph.get_full_solution_cost() < min_cost:
            best_graph = graph
            min_cost = cost
    return best_graph


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


def run_inference(model, graphs: List[Tuple[GraphCollection, List[int]]]) -> List[GraphCollection]:
    """Runs inference over the given graphs and returns a list of them solved
    """
    solved_graphs = []
    for g, sol in graphs:
        g = deepcopy(g)
        g.update_node(sol[1][0]) # unf we have to do this for now
        res_g = greedy_inference_single(model, g)
        solved_graphs.append(res_g)
    return solved_graphs

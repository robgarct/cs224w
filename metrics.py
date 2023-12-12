import torch
import numpy as np
from torch_geometric.data import Batch
from typing import Dict, Any, List

class Metric():
    def __init__(self, val):
        self.val = val
    
    def mean(self, metrics: List["Metric"]) -> "Metric":
        return np.mean(list(map(lambda x: x.val, metrics)))

class KeyedMetric(Metric):
    def mean(self, metrics: List["KeyedMetric"]) -> "KeyedMetric":
        vals = {}
        for metric in metrics:
            for k, v in metric.val.items():
                if not k in vals: vals[k] = []
                vals[k].append(v)
        means = {}
        for k, v in vals.items():
            means[k] = np.mean(v)
        return means


def compute_accuracy(logits: torch.FloatTensor, graphs: Batch) -> Metric:
    """Returns a metric where the metric.val is the mean accuracy.
    """
    nexts = graphs.y.reshape(-1)
    preds = torch.argmax(logits, dim=-1).reshape(-1)
    assert nexts.shape == preds.shape
    acc = (torch.argmax(logits) == nexts).to(torch.float32)
    return Metric(acc.mean().cpu().detach().item())

def compute_accuracy_per_rem_nodes(
    logits: torch.FloatTensor,
    graphs: Batch,
    buckets=4
) -> KeyedMetric:
    """Returns a keyed metric, where metric.val is a dict mapping
    remaining_nodes_count -> mean_accuracy.
    """

    nexts = graphs.y.reshape(-1)
    preds = torch.argmax(logits, dim=-1).reshape(-1)
    assert nexts.shape == preds.shape
    
    acc = (torch.argmax(logits) == nexts).to(torch.float32)
    # keeps track of how many nodes are remaining in the given solution
    rem_nodes_cnt = graphs.remaining_nodes_cnt
    assert acc.shape == rem_nodes_cnt.shape
    # TODO(roberto): make this faster
    metrics = []
    for k,v in zip(rem_nodes_cnt, acc):
        metrics.append(KeyedMetric({k.item():v}))
    
    return KeyedMetric(metrics[0].mean(metrics))
    
def compute_metrics(logits: torch.FloatTensor, graphs: Batch) -> Dict[str, Metric]:
    return {
        "Accuracy": compute_accuracy(logits, graphs),
        "Accuracy Breakdown": compute_accuracy_per_rem_nodes(logits, graphs)
    }
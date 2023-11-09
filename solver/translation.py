import os
from subprocess import check_call
import time
import sys
import pickle
import numpy as np
import torch
from tqdm import tqdm
from functools import reduce
import pandas as pd
from baseline_lkh3 import CAPACITIES

def format_and_save(val_set, lkh_costs, lkh_paths, dir):
    """
    - instances[i][0] -> depot(x, y)
    - instances[i][1] -> nodes(x, y) * samples
    - instances[i][2] -> nodes(demand) * samples
    - instances[i][3] -> capacity (of vehicle) (should be the same for all in theory)

    - sols[i][0] -> cost
    - sols[i][1] -> path (doesn't include depot at the end)
    """
    depo, graphs, demand = val_set
    num_samples, graph_size, _ = graphs.shape
    instances = [[depo[i].tolist(), graphs[i].tolist(), demand[i].tolist(), float(CAPACITIES[graph_size])] for i in range(len(depo))]
    sols = list(zip(lkh_costs, lkh_paths))
    assert len(sols) == len(instances)
    graphs = list(zip(instances, sols))
    pd.to_pickle(graphs, f'{dir}/graphs_solved.pkl')
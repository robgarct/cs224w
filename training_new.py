"""
Training utilities
"""
import sys

from torch.optim import Adam
from tqdm import tqdm
import torch
from torch import nn
import itertools
import pandas as pd

from model import Model
from generator import *

from torch_geometric.loader import DataLoader
from typing import List
from util import *

import pickle5 as pkl


def get_solution_instances(graphs_path: str) -> List[SolutionInstance]:
    """Given the path to the pickle of graphs generated and further solved
    using LKH, returns a list of SolutionInstances.
    """
    with open(graphs_path, "rb") as f:
        graphs = pkl.load(f)
    graphs = [create_graph_solver_solution(g, sol) for g, sol in graphs]

    return graphs


def get_data_loaders(graphs_path: str, train_split_size=0.8, batch_size=16) -> DataLoader:
    """Given the path to the pickle of graphs generated and further solved
    using LKH, returns 2 PYG's DataLoaders, one for training and the other
    for validation.
    """
    graphs = []
    sol_instances = get_solution_instances(graphs_path)
    split_idx = int(train_split_size * len(sol_instances))

    for sol_instance in sol_instances:
        ## need to specify how many partial solutions we need partial solution index
        #cvrp_graphs = sol_instance.get_partial_solutions()
        cvrp_graphs = sol_instance.get_k_partial_solutions(2)
        # convert to pytorch geometric dataset
        cvrp_graphs = list(map(lambda g: g.export_pyg(), cvrp_graphs))
        graphs.append(cvrp_graphs)

    return graphs
    #sys.exit()


    train_graphs = itertools.chain.from_iterable(graphs[:split_idx])
    valid_graphs = itertools.chain.from_iterable(graphs[split_idx:])

    train_graphs = DataLoader(train_graphs, shuffle=True, batch_size=batch_size)
    valid_graphs = DataLoader(valid_graphs, shuffle=True, batch_size=batch_size)
    return train_graphs, valid_graphs


def compute_accuracy(logits, nexts):
    """Given logit predictions and the ground truth nodes to be visited next,
    computes the accuracy of the model.
    """
    return (torch.argmax(logits) == nexts).mean().cpu().detach()


def eval(model: Model, data_loder: DataLoader):
    """Evaluates the model on the given data loader.
    """
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    accs = []
    for batched_graphs in data_loder:
        logits = model(batched_graphs)
        nexts = batched_graphs.y
        loss = loss_fn(logits, nexts)
        valid_acc = compute_accuracy(logits, nexts)
        losses.append(loss.cpu().detach())
        accs.append(valid_acc.cpu().detach())
    return torch.tensor(losses).mean().item(), torch.tensor(accs).mean().item()


def train(model: Model, graphs_path: str, epochs: int = 20, batch_size: int = 16, learning_rate: float = 3e-5,
          eval_epochs: int = 1):
    """Runs training for the model on the graphs pointed out by the given graphs path.
    The model weights are updated inplace, so this function returns nothing.
    """
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_dl, valid_dl = get_data_loaders(graphs_path, batch_size=batch_size)

    for e in tqdm(range(epochs), "Epochs", epochs):
        for batched_graphs in train_dl:
            nexts = batched_graphs.y
            # this should return very small numbers for irrelevant classes
            logits = model(batched_graphs)  # (batch_size, classes)

            loss = loss_fn(logits, nexts)
            with torch.no_grad():
                acc = compute_accuracy(logits, nexts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Train {e} - loss:{loss:.4f}, acc:{acc:.4f}")

            if e % eval_epochs == 0:
                # run eval
                model.eval()
                with torch.no_grad():
                    valid_loss, valid_acc = eval(model, valid_dl)
                model.train()
                print(f"Valid {e} - loss:{valid_loss:.4f}, acc:{valid_acc:.4f}")

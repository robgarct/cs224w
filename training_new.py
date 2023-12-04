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

# import pickle5 as pkl


def get_solution_instances(graphs_path: str, max_instances: int) -> List[GraphCollection]:
    """Given the path to the pickle of graphs generated and further solved
    using LKH, returns a list of SolutionInstances.
    """
    graphs = pd.read_pickle(graphs_path)
    if max_instances != -1:
        graphs = [create_graph_solver_solution(g, sol) for g, sol in graphs[:max_instances]]
    else:
        graphs = [create_graph_solver_solution(g, sol) for g, sol in graphs]
    return graphs


def get_data_loaders(graphs_path: str, train_split_size=0.8, batch_size=16, max_instances=-1) -> DataLoader:
    """Given the path to the pickle of graphs generated and further solved
    using LKH, returns 2 PYG's DataLoaders, one for training and the other
    for validation.
    """
    graphs = []
    sol_instances = get_solution_instances(graphs_path, max_instances)
    split_idx = int(train_split_size * len(sol_instances))

    for sol_instance in tqdm(sol_instances, "Parsing Graphs"):
        ## need to specify how many partial solutions we need partial solution index
        #cvrp_graphs = sol_instance.get_partial_solutions()
        cvrp_graphs = sol_instance.get_all_graphs()
        # convert to pytorch geometric dataset
        cvrp_graphs = list(map(lambda g: g.export_pyg(), cvrp_graphs))
        graphs.append(cvrp_graphs)

    train_graphs = itertools.chain.from_iterable(graphs[:split_idx])
    valid_graphs = itertools.chain.from_iterable(graphs[split_idx:])

    train_graphs = DataLoader(list(train_graphs), shuffle=True, batch_size=batch_size)
    valid_graphs = DataLoader(list(valid_graphs), shuffle=True, batch_size=batch_size)
    return train_graphs, valid_graphs


def compute_accuracy(logits, nexts):
    """Given logit predictions and the ground truth nodes to be visited next,
    computes the accuracy of the model.
    """
    return (torch.argmax(logits, dim=-1) == nexts).to(torch.float32).mean().cpu().detach()


def eval(model: Model, data_loder: DataLoader):
    """Evaluates the model on the given data loader.
    """
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    accs = []
    for batched_graphs in data_loder:
        batched_graphs = batched_graphs.cuda()
        logits = model(batched_graphs)
        nexts = batched_graphs.y
        loss = loss_fn(logits, nexts)
        valid_acc = compute_accuracy(logits, nexts)
        losses.append(loss.cpu().detach())
        accs.append(valid_acc.cpu().detach())
    return torch.tensor(losses).mean().item(), torch.tensor(accs).mean().item()


def train(model: Model, graphs_path: str, epochs: int = 20, batch_size: int = 16, learning_rate: float = 3e-5,
          eval_epochs: int = 1, max_instances: int = -1):
    """Runs training for the model on the graphs pointed out by the given graphs path.
    The model weights are updated inplace, so this function returns nothing.
    """
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_dl, valid_dl = get_data_loaders(graphs_path, batch_size=batch_size, max_instances=max_instances)

    for e in range(epochs):
        epoch_acc = 0
        epoch_loss = 0
        n_batches = len(train_dl)
        for batched_graphs in tqdm(train_dl, "Batches", n_batches):
            batched_graphs = batched_graphs.cuda()
            nexts = batched_graphs.y
            # this should return very small numbers for irrelevant classes
            logits = model(batched_graphs)  # (batch_size, classes)

            loss = loss_fn(logits, nexts)

            with torch.no_grad():
                acc = compute_accuracy(logits, nexts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.cpu().detach() / n_batches
            epoch_acc += acc.cpu().detach() / n_batches

        print(f"Train {e} - loss:{loss:.4f}, acc:{acc:.4f}")

        if e % eval_epochs == 0:
            # run eval
            model.eval()
            with torch.no_grad():
                valid_loss, valid_acc = eval(model, valid_dl)
            model.train()
            print(f"Valid {e} - loss:{valid_loss:.4f}, acc:{valid_acc:.4f}")

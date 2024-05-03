import argparse
import os
from collections import defaultdict
from typing import Callable, Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

from src.utils.eval import getBinaryDCG, getHR, getRR
from src.utils.tools.tools import ROOT_PATH, create_checkpoint_folder, get_parent_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run context Aware NCF.")
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="checkpoints/CNCF_I",
        type=str,
        metavar="PATH",
        help="checkpoint directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/CNCF/YELP/yelp-1.yaml",
        help="Path to the config file.",
    )
    opts = parser.parse_args()
    return opts


_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_device_ids = [0] if torch.cuda.is_available() else None
_optimizers = {"adam": optim.Adam, "SGD": optim.SGD}
_loss_fn = {"BCE": nn.BCELoss(), "MSE": nn.MSELoss()}


def setup(rank, world_size):
    """
    Function to setup the distributed training environment

    Args:
        - rank: Rank of the current process
        - world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    print(
        f"Training on rank {rank} with PID {os.getpid()} on CPU core {os.sched_getaffinity(0)}"
    )

    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_epoch(
    optimizer: Optimizer,
    loss_fn: Callable,
    train_loader: DataLoader,
    model: Module,
    losses: Dict,
):
    """
    Function that performs a training epoch step.

    Args:
        - oprimizer: The optimizer for the model
        - loss_fn: the loss function from the model
        - train_loader: Train dataloader
        - model: Model initialized

    """
    global _device

    idx_loss = len(losses.keys())
    model.train()
    total_loss = 0.0
    num_batches = 0

    # calculate_memory_allocation()
    for batch in train_loader:
        user_input = batch["user"].to(_device)
        item_input = batch["item"].to(_device)
        context_input = batch["context"].to(_device)
        ratings = batch["rating"].to(_device)
        ratings = ratings.view(-1, 1)

        output = model(user_input, item_input, context_input)
        loss = loss_fn(output, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        # if num_batches % 100 == 0:
        #     logger.log(f"Batch {num_batches} - Loss: {loss.item()}")

    losses[idx_loss] = total_loss / num_batches


def evaluate_model(model_pos, data_loader, topK: int):
    global _device
    # Set Model to evaluation
    model_pos.eval()

    # Initialize containers for users, items, and predictions
    all_users = []
    all_items = []
    all_predictions = []
    all_gtItems = []

    with torch.no_grad():
        for batch in data_loader:
            user_input = batch["user"].to(_device)
            item_input = batch["item"].to(_device)
            gtItems = batch["gtItem"]
            context_input = batch["context"].to(_device)
            ratings = batch["rating"].to(_device)
            ratings = ratings.view(-1, 1)

            batch_predictions = model_pos(user_input, item_input, context_input)

            all_predictions.append(batch_predictions.cpu().numpy())
            all_users.append(user_input.cpu().numpy())
            all_items.append(item_input.cpu().numpy())
            all_gtItems.append(gtItems.numpy())

    # Concatenate all arrays into single NumPy arrays
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()
    all_users = np.concatenate(all_users, axis=0).flatten()
    all_items = np.concatenate(all_items, axis=0).flatten()
    all_gtItems = np.concatenate(all_gtItems, axis=0).flatten()

    # Initialize a defaultdict to store lists of (item, score) tuples for each user
    user_predictions = defaultdict(list)

    for user, item, score, gtItem in zip(
        all_users, all_items, all_predictions, all_gtItems
    ):
        user_predictions[user].append((item, score, gtItem))

    hrs, rrs, ndcgs = [], [], []
    for user, items_scores in user_predictions.items():
        # Sort items based on scores in descending order and select top-K
        topK_items = sorted(items_scores, key=lambda x: x[1], reverse=True)[:topK]
        gtItem = topK_items[0][2]
        topK_items = [item for item, score, gt in topK_items]

        # Evaluation
        hrs.append(getHR(topK_items, [gtItem]))
        rrs.append(getRR(topK_items, [gtItem]))
        ndcgs.append(getBinaryDCG(topK_items, [gtItem]))

    return np.mean(hrs), np.mean(rrs), np.mean(ndcgs)


def train(rank,args, opts):
    global _optimizers
    global _loss_fn
    global _device
    global logger

    setup(rank, args.world_size)

    # Folder structure checkpoint
    data_name, check_point_path = create_checkpoint_folder(args, opts)
    parent_path = get_parent_path(ROOT_PATH)
    processed_data_path = os.path.join(parent_path, args.processed_data_root)

    model = nn.Sequential(nn.Linear(10, 10))  # Example simple model
    model = DDP(model, device_ids=None)  # Use CPUs

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Example data: just creating random tensors
    dataset = [torch.randn(10, 10) for _ in range(100)]
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

    for epoch in range(2):
        sampler.set_epoch(epoch)
        for data in dataloader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)  # Simple loss for example
            loss.backward()
            optimizer.step()

    cleanup()


if __name__ == "__main__":
    world_size = min(4, os.cpu_count())  # Use up to 4 cores or the max available
    print(f"Running DDP with {world_size} processes")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

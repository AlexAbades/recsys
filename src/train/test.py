import argparse
import os
from collections import defaultdict
from time import time
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

from src.data.cncf_collate_fn import cncf_negative_sampling
from src.data.cncf_interaction_dataset import CNCFDataset
from src.models.CNCF.cncf import ContextualNeuralCollavorativeFiltering
from src.utils.eval import getBinaryDCG, getHR, getRR
from src.utils.model_stats.stats import save_accuracy, save_checkpoint
from src.utils.tools.tools import (
    ROOT_PATH,
    create_checkpoint_folder,
    get_config,
    get_parent_path,
)


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


# _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# _device_ids = [0] if torch.cuda.is_available() else None

_device = torch.device("cpu")
_device_ids = [0] if _device.type == "cuda" else None
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
    pid = os.getpid()
    print(f"The current process ID is: {pid}")
    c = 0 
    # calculate_memory_allocation()
    for batch in train_loader:
        user_input = batch["user"].to(_device)
        item_input = batch["item"].to(_device)
        context_input = batch["context"].to(_device)
        ratings = batch["rating"].to(_device)
        ratings = ratings.view(-1, 1)
        if not c:
            print(f"User Input: {user_input.shape}")
            c += 1

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


def evaluate_model(rank, model_pos, data_loader, losses, topK: int):
    global _device
    global best_hr, best_mrr, best_ndcg
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

    hrs = np.mean(hrs)
    rrs = np.mean(rrs)
    ndcgs = np.mean(ndcgs)

    local_metrics = torch.tensor([hrs, rrs, ndcgs], dtype=torch.float32)
    gathered_metrics = [torch.zeros_like(local_metrics) for _ in range(4)]
    dist.all_gather(gathered_metrics, local_metrics)
    
    if dist.get_rank() == 0:
        metrics_tensor = torch.stack(gathered_metrics)
        average_metrics = torch.mean(metrics_tensor, dim=0)
        hrs, rrs, ndcgs = (
            average_metrics[0].item(),
            average_metrics[1].item(),
            average_metrics[2].item(),
        )
        if bool(losses.items()):
            epoch = list(losses.items())[-1][0]
            loss = list(losses.items())[-1][1]
        else:
            epoch = 0
            loss = 0.0
        print(
                f"HR = {hrs:.4f}, MRR = {rrs:.4f}, NDCG = {ndcgs:.4f}, loss = {loss:.4f}"
            )
        if hrs > best_hr:
            best_hr = hrs
            best_mrr = rrs
            best_ndcg = ndcgs

            torch.save(model_pos.state_dict(), chk_path_best)
            print("Saving best model into: ", chk_path_best)
            save_accuracy(
                chk_path_best,
                hr=best_hr,
                mrr=best_mrr,
                ndcg=best_ndcg,
                epoch=epoch,
            )
    return np.mean(hrs), np.mean(rrs), np.mean(ndcgs)


def train_with_config(rank, world_size, args, opts):
    global _optimizers
    global _loss_fn
    global _device
    global _device_ids
    global best_hr, best_mrr, best_ndcg
    global chk_path_best
    global logger

    setup(rank, world_size)

    # Folder structure checkpoint
    data_name, check_point_path = create_checkpoint_folder(args, opts)
    parent_path = get_parent_path(ROOT_PATH)
    processed_data_path = os.path.join(parent_path, args.processed_data_root)

    # Save checkpoint
    chk_path_latest = os.path.join(check_point_path, "latest_epoch.bin")
    chk_path_best = os.path.join(check_point_path, "best_epoch.bin")

    # Load the data
    train_data = CNCFDataset(
        processed_data_path,
        split="train",
        n_items=args.num_items,
        num_negative_samples=args.num_negative_instances_train,
    )
    print(train_data.data.shape)
    test_data = CNCFDataset(
        processed_data_path,
        split="test",
        n_items=args.num_items,
        num_negative_samples=args.num_negative_instances_test,
    )
    # print(test_data.data.shape)

    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank)
    # print instances per sampler 
    print(f"Rank: {rank} - Train Sampler: {len(train_sampler)} - Test Sampler: {len(test_sampler)}")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,  # Do we really need this?
        collate_fn=cncf_negative_sampling,
    )
    # print instances per loader
    print(f"Rank: {rank} - Train Loader: {len(train_loader)}")
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.num_workers,  # Do we really need this?
        collate_fn=cncf_negative_sampling,
    )

    # Num User, Items Context Features
    num_users = args.num_users
    num_items = args.num_items
    num_context = args.num_context

    # Model
    model = ContextualNeuralCollavorativeFiltering(
        num_users=num_users,
        num_items=num_items,
        num_context=num_context,
        mf_dim=args.num_factors,
        layers=args.layers,
    )
    model = DDP(model, device_ids=None)
    best_hr = 0.0
    best_mrr = 0.0
    best_ndcg = 0.0
    # Loss Function
    loss_fn = _loss_fn[args.loss]

    # Optimizer
    optimizer = _optimizers[args.optimizer](model.parameters(), lr=args.lr)

    losses = dict()
    min_loss = float("inf")

    # Initialize performance

    evaluate_model(rank, model, test_loader, losses, topK=args.topK)
    # print(f"Init: HR = {hr:.4f}, MRR = {mrr:.4f}, NDCG = {ndcg:.4f}")

    for epoch in range(args.epochs):
        print(f"Rank: {rank} and epoch {epoch}: test print")
        start_time = time()


        # Curriculum Learning
        train_epoch(optimizer, loss_fn, train_loader, model, losses)

        # Sample Train Time
        train_time = (time() - start_time) / 60

        if not (epoch % args.verbose):
            evaluate_model(rank, model, test_loader, losses, topK=args.topK)
            test_time = ((time() - start_time) / 60) - train_time
            total_time = train_time + test_time

        if dist.get_rank() == 0:
            print(
                f"[{epoch:d}] Elapsed time: {total_time:.2f}m - Train time: {train_time:.2f}m - Test time: {test_time:.2f}"
            )
            # Save lastest model
            save_checkpoint(
                chk_path_latest, epoch, args.lr, optimizer, model, min_loss
            )
            save_accuracy(
                check_point_path + "/latest_epoch",
                hr=best_hr,
                mrr=best_mrr,
                ndcg=best_ndcg,
                epoch=epoch,
            )
            # print(
            #     f"HR = {hr:.4f}, MRR = {mrr:.4f}, NDCG = {ndcg:.4f}, loss = {losses[epoch]:.4f}"
            # )

    cleanup()


if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    world_size = min(args.cpu_nodes, os.cpu_count())
    print(f"Running DDP with {world_size} processes")
    mp.spawn(
        train_with_config, args=(world_size, args, opts), nprocs=world_size, join=True
    )

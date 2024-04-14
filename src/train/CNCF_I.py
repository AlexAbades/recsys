import argparse
import errno
import os
from collections import defaultdict
from time import time
from typing import Callable, Dict

import numpy as np
import torch
from torch import nn, optim
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


from src.data.cncf_collate_fn import cncf_negative_sampling
from src.data.cncf_interaction_dataset import CNCFDataset
from src.models.CNCF.cncf import CNCF
from src.utils.eval import getBinaryDCG, getHR, getRR
from src.utils.model_stats.stats import (
    calculate_model_size,
    save_accuracy,
    save_checkpoint,
)
from src.utils.tools.tools import (
    ROOT_PATH,
    TextLogger,
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


_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Maybe extract as dictionary
_optimizers = {"adam": optim.Adam, "SGD": optim.SGD}
_loss_fn = {"BCE": nn.BCELoss(), "MSE": nn.MSELoss()}


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

    losses[idx_loss] = loss


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


def train_with_config(args, opts):
    global _optimizers
    global _loss_fn
    global _device

    # Folder structure checkpoint
    data_name, check_point_path = create_checkpoint_folder(args, opts)
    # processed_data_path = os.chdir(ROOT_PATH, args.processed_data_root)
    log_path = os.path.join(ROOT_PATH, f"logs/logs_{args.foldername}")
    # go up a folder
    parent_path = get_parent_path(ROOT_PATH)
    processed_data_path = os.path.join(parent_path, args.processed_data_root)

    logger = TextLogger(log_path)

    print(f"Running in device: {_device}")
    logger.log(f"Running in device: {_device}")

    # Load preprocessed Data
    train_data = CNCFDataset(
        processed_data_path,
        split="train",
        n_items=args.num_items,
        num_negative_samples=5,
    )
    logger.log(f"Train Data Loaded")
    test_data = CNCFDataset(
        processed_data_path,
        split="test",
        n_items=args.num_items,
        num_negative_samples=99,
    )
    logger.log(f"Test Data Loaded")

    # Dataloader
    train_loader = DataLoader(
        train_data, args.batch_size, collate_fn=cncf_negative_sampling
    )
    test_loader = DataLoader(
        test_data, args.batch_size, collate_fn=cncf_negative_sampling
    )

    # Num User, Items Context Features
    num_users = 528685
    num_items = args.num_items
    num_context = 22

    model = CNCF(
        num_users=num_users,
        num_items=num_items,
        num_context=num_context,
        mf_dim=args.num_factors,
        layers=args.layers,  # Has to be
    ).to(_device)
    logger.log(calculate_model_size(model))

    # Initialize Optimizer and Loss function
    loss_fn = _loss_fn[args.loss]
    optimizers = _optimizers[args.optimizer](model.parameters(), lr=args.lr)
    losses = dict()
    min_loss = float("inf")

    # Initialize performance
    (hr, mrr, ndcg) = evaluate_model(model, test_loader, topK=args.topK)
    print(f"Init: HR = {hr:.4f}, MRR = {mrr:.4f}, NDCG = {ndcg:.4f}")
    best_hr = hr

    for epoch in range(args.epochs):
        print("Training epoch %d." % epoch)
        start_time = time()
        # TODO: We have to actualize in each epoch the data.
        # Curriculum Learning
        train_epoch(optimizers, loss_fn, train_loader, model, losses)

        # Sample Train Time
        train_time = (time() - start_time) / 60

        # Update Losses
        if losses[epoch] < min_loss:
            min_loss = losses[epoch]
        if losses[epoch] < min_loss:
            min_loss = losses[epoch]

        # Eval model
        if not (epoch % args.verbose):
            (hr, mrr, ndcg) = evaluate_model(model, test_loader, topK=args.topK)
            test_time = ((time() - start_time) / 60) - train_time
            total_time = train_time + test_time

            print(
                f"[{epoch:d}] Elapsed time: {total_time:.2f}m - Train time: {train_time:.2f}m - Test time: {test_time:.2f}"
            )
            print(
                f"HR = {hr:.4f}, MRR = {mrr:.4f}, NDCG = {ndcg:.4f}, loss = {losses[epoch]:.4f}"
            )

            # Save checkpoints
            chk_path = os.path.join(check_point_path, "epoch_{}.bin".format(epoch))
            chk_path_latest = os.path.join(check_point_path, "latest_epoch.bin")
            chk_path_best = os.path.join(
                check_point_path, "best_epoch.bin".format(epoch)
            )

            # Save lastest model
            save_checkpoint(
                chk_path_latest, epoch, args.lr, optimizers, model, min_loss
            )
            save_accuracy(
                check_point_path + "/latest_epoch",
                hr=hr,
                mrr=mrr,
                ndcg=ndcg,
                epoch=epoch,
            )

            # Save best Model based on HR
            if hr < best_hr:
                best_hr = hr
                save_checkpoint(
                    chk_path_best, epoch, args.lr, optimizers, model, min_loss
                )
                save_accuracy(
                    check_point_path + "/best_epoch",
                    hr=hr,
                    mrr=mrr,
                    ndcg=ndcg,
                    epoch=epoch,
                )


if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_with_config(args, opts)

import argparse
import os
from ast import Dict
from collections import defaultdict
from time import time
from typing import Callable

import numpy as np
import torch
from torch import nn, optim
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.data.ContextInteractionDataLoader import ContextInteractionDataLoader
from src.models.AutoEncoder.AE import AutoEncoder
from src.models.contextNFC.context_nfc import DeepNCF
from src.utils.eval import getBinaryDCG, getHR, getRR
from src.utils.model_stats.stats import (
    load_model_with_params,
    save_accuracy,
    save_checkpoint,
    save_model_with_params,
)
from src.utils.tools.tools import ROOT_PATH, create_checkpoint_folder, get_config

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Maybe extract as dictionary
_optimizers = {"adam": optim.Adam, "SGD": optim.SGD}
_loss_fn = {"BCE": nn.BCELoss(), "MSE": nn.MSELoss()}


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
        default="configs/CNCF_AE/FRAPPE/frappe1.yaml",
        metavar="PATH",
        help="Path to the config file.",
    )
    parser.add_argument(
        "-ae",
        "--ae_path",
        type=str,
        default="checkpoints/AE/FRAPPE/RealData/frap-no-init-weights-b256-2/best_epoch.bin",
        metavar="PATH",
        help="Path to the autoencoder",
    )
    opts = parser.parse_args()
    return opts


def train_epoch(
    optimizer: Optimizer,
    loss_fn: Callable,
    train_loader: DataLoader,
    rs_model: Module,
    ae_model: Module,
    losses: Dict,
    device: torch.device,  # Added device as a parameter for clarity
):
    """
    Performs a training epoch step for a recommendation system, utilizing an autoencoder for contextual feature transformation.

    Args:
        optimizer: The optimizer for the RS model.
        loss_fn: The loss function for the RS model.
        train_loader: DataLoader for the training data.
        rs_model: The recommendation system model.
        ae_model: The autoencoder model for contextual feature transformation.
        losses: A dictionary to store losses.
        device: The device (CPU/GPU) on which to perform the computation.
    """
    rs_model.train()
    ae_model.eval()  # Ensure AE model is in eval mode
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        user_input, item_input, context_input, ratings = (
            batch["user"].to(device),
            batch["item"].to(device),
            batch["context"].to(device),
            batch["rating"].to(device).view(-1, 1),
        )

        with torch.no_grad():
            latent = ae_model(context_input)["latent"]

        output = rs_model(user_input, item_input, latent)
        loss = loss_fn(output, ratings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    losses[len(losses)] = avg_loss  # Store the average loss for the epoch


def evaluate_model(
    rs_model: Module, ae_model: Module, data_loader: DataLoader, topK: int
):
    global _device
    # Set Model to evaluation
    rs_model.eval()
    ae_model.eval()

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

            latent = ae_model(context_input)["latent"]
            batch_predictions = rs_model(user_input, item_input, latent)

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
    processed_data_path = os.path.join(ROOT_PATH, args.processed_data_root)
    ae_model_path = os.path.join(ROOT_PATH, opts.ae_path)

    print(f"Running in device: {_device}")

    # Load preprocessed Data
    train_data = ContextInteractionDataLoader(processed_data_path, split="train")
    test_data = ContextInteractionDataLoader(
        processed_data_path,
        split="test",
        num_negative_samples=99,
    )

    # Dataloader
    train_loader = DataLoader(train_data, args.batch_size)
    test_loader = DataLoader(test_data, args.batch_size)

    # Num User, Items Context Features
    num_users = train_data.num_users
    num_items = train_data.num_items
    num_context = train_data.num_context

    ae_model = load_model_with_params(ae_model_path, AutoEncoder).to(_device)

    rs_model = DeepNCF(
        num_users=num_users,
        num_items=num_items,
        num_context=args.ae_bottleneck,
        mf_dim=args.num_factors,
        layers=args.layers,
    ).to(_device)

    # Initialize Optimizer and Loss function
    loss_fn = _loss_fn[args.loss]
    optimizer = _optimizers[args.optimizer](rs_model.parameters(), lr=args.lr)
    losses = dict()
    min_loss = float("inf")

    # Initialize performance
    (hr, mrr, ndcg) = evaluate_model(
        rs_model=rs_model, ae_model=ae_model, data_loader=test_loader, topK=args.topK
    )
    print(f"Init: HR = {hr:.4f}, MRR = {mrr:.4f}, NDCG = {ndcg:.4f}")
    best_hr = hr

    for epoch in range(args.epochs):
        print("Training epoch %d." % epoch)
        start_time = time()
        # TODO: We have to actualize in each epoch the data.
        # Curriculum Learning
        train_epoch(
            optimizer,
            loss_fn,
            train_loader,
            rs_model,
            ae_model,
            losses=losses,
            device=_device,
        )

        # Sample Train Time
        train_time = (time() - start_time) / 60

        # Update Losses
        if losses[epoch] < min_loss:
            min_loss = losses[epoch]
        if losses[epoch] < min_loss:
            min_loss = losses[epoch]

        # Eval model
        if not (epoch % args.verbose):
            (hr, mrr, ndcg) = evaluate_model(
                rs_model=rs_model,
                ae_model=ae_model,
                data_loader=test_loader,
                topK=args.topK,
            )
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
                chk_path_latest, epoch, args.lr, optimizer, rs_model, min_loss
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
                    chk_path_best, epoch, args.lr, optimizer, rs_model, min_loss
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

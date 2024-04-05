import argparse
import errno
import os
from time import time
from typing import Callable, Dict

import numpy as np
import torch
from torch import nn, optim
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from src.data.ContextRatingDataLoader import ContextRatingDataLoader
from src.models.contextNFC.context_nfc import DeepNCF
from src.utils.eval import mean_absolute_error, root_mean_squared_error
from src.utils.model_stats.stats import save_accuracy, save_checkpoint
from src.utils.tools.tools import (ROOT_PATH, create_checkpoint_folder,
                                   get_config)


def parse_args():
    parser = argparse.ArgumentParser(description="Run context Aware NCF.")
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="checkpoints/context_NFC",
        type=str,
        metavar="PATH",
        help="checkpoint directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/CNCF_R/frappe-1.yaml",
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


def train_with_config(args, opts):
    global _optimizers
    global _loss_fn
    global _device

    # Folder structure checkpoint
    data_name, check_point_path = create_checkpoint_folder(args, opts)
    processed_data_path = os.path.join(ROOT_PATH, args.processed_data_root, data_name)
    print(processed_data_path)
    # Ensure Device
    print(f"Running in device: {_device}")

    # Load preprocessed Data
    train_data = ContextRatingDataLoader(processed_data_path + ".train.rating")
    test_data = ContextRatingDataLoader(processed_data_path + ".test.rating")

    # Dataloader
    train_loader = DataLoader(train_data, args.batch_size)
    test_loader = DataLoader(test_data, args.batch_size)

    # Num User, Items Context Features
    num_users = train_data.num_users
    num_items = train_data.num_items
    num_context = train_data.num_context

    # Initialize the model
    model = DeepNCF(
        num_users=num_users,
        num_items=num_items,
        num_context=num_context,
        mf_dim=args.num_factors,
        layers=args.layers,
        binary_classification=False,
    ).to(_device)

    # Initialize Optimizer and Loss function
    loss_fn = _loss_fn[args.loss]
    optimizers = _optimizers[args.optimizer](model.parameters(), lr=args.lr)
    losses = dict()
    min_loss = float("inf")

    # Initialize performance
    (rmse, mae) = evaluate_model(model, test_loader)
    print("Init: RMSE = %.4f, MAE = %.4f" % (rmse, mae))
    best_rmse, best_mae, best_iter = rmse, mae, -1

    for epoch in range(args.epochs):
        print("Training epoch %d." % epoch)
        start_time = time()

        # Curriculum Learning
        train_epoch(optimizers, loss_fn, train_loader, model, losses)

        # Sample Train Time
        train_time = (time() - start_time) / 60

        # Update
        if losses[epoch] < min_loss:
            min_loss = losses[epoch]
        if losses[epoch] < min_loss:
            min_loss = losses[epoch]

        # Eval model
        if not (epoch % args.verbose):
            (rmse, mae) = evaluate_model(model, test_loader)
            test_time = ((time() - start_time) / 60) - train_time
            total_time = train_time + test_time

            print(
                "[%d] time %.2f m - RMSE = %.4f, MAE = %.4f, loss = %.4f. --- Train time %.2f m - Test time %.2f m"
                % (
                    epoch,
                    total_time,
                    rmse,
                    mae,
                    losses[epoch],
                    train_time,
                    test_time,
                )
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
                check_point_path + "/latest_epoch", rmse=rmse, mae=mae, epoch=epoch
            )

            # Save best Model based on RSME
            if rmse < best_rmse:
                best_rmse = rmse
                save_checkpoint(
                    chk_path_best, epoch, args.lr, optimizers, model, min_loss
                )
                save_accuracy(
                    check_point_path + "/best_epoch", rmse=rmse, mae=mae, epoch=epoch
                )


def evaluate_model(model_pos, data_loader):
    global _device
    # Set Model to evaluation
    model_pos.eval()

    rsme_all = []
    mae_all = []
    with torch.no_grad():
        for batch in data_loader:
            user_input = batch["user"].to(_device)
            item_input = batch["item"].to(_device)
            context_input = batch["context"].to(_device)
            ratings = batch["rating"].to(_device)
            ratings = ratings.view(-1, 1)

            predictions = model_pos(user_input, item_input, context_input)

            # Evaluate
            rsme = root_mean_squared_error(predictions, ratings)
            mae = mean_absolute_error(predictions, ratings)
            rsme_all.append(rsme)
            mae_all.append(mae)

    return np.mean(rsme_all), np.mean(mae_all)


if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_with_config(args, opts)

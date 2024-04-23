import argparse
import os
from time import time
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from src.data.cncf_rating_datset import ContextRatingDataLoader
from src.models.AutoEncoder.AE import AutoEncoder
from src.utils.eval import mean_absolute_error, root_mean_squared_error
from src.utils.model_stats.stats import (
    plot_and_save_dict,
    save_accuracy,
    save_dict_to_file,
    save_model_specs,
    save_model_with_params,
    save_opts_args,
)
from src.utils.tools.tools import ROOT_PATH, create_checkpoint_folder, get_config, get_parent_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run context Aware NCF.")
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="checkpoints/AE",
        type=str,
        metavar="PATH",
        help="checkpoint directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/AE/frappe.yaml",
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
    use_random_data: bool = False,
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
        # context_input = batch["context"].to(_device)
        context_input = batch[0].to(_device)

        if use_random_data:
            context_input = create_noise_tensor(context_input)

        output = model(context_input)
        loss = loss_fn(output["prediction"], context_input)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    losses[idx_loss] = avg_loss


def create_noise_tensor(context_input):
    noise_tensor = torch.randint(
        0, 2, size=context_input.shape, dtype=torch.float32, device=_device
    )
    noise = torch.rand(context_input.shape[0], 1)
    noise_tensor[:, 0] = noise.squeeze()
    return noise_tensor


def evaluate_model(model_pos, data_loader):
    global _device
    # Set Model to evaluation
    model_pos.eval()

    rsme_all = []
    mae_all = []
    with torch.no_grad():
        for batch in data_loader:
            # context_input = batch.to(_device)
            context_input = batch[0].to(_device)

            output = model_pos(context_input)

            # Evaluate
            rsme = root_mean_squared_error(
                predictions=output["prediction"], ground_truth=context_input
            )
            mae = mean_absolute_error(
                predictions=output["prediction"], ground_truth=context_input
            )
            rsme_all.append(rsme)
            mae_all.append(mae)

    return np.mean(rsme_all), np.mean(mae_all)


def train_with_config(args, opts):
    global _optimizers
    global _loss_fn
    global _device

    s1  = time()

    # Folder structure checkpoint
    data_name, check_point_path = create_checkpoint_folder(args, opts)
    # processed_data_path = os.path.join(ROOT_PATH, args.processed_data_root)

    print(f"Running in device: {_device}")
    print(f"Batch size: {args.batch_size}, lr: {args.lr}")

    # Load preprocessed Data
    parent_path = get_parent_path(ROOT_PATH)
    processed_data_path = os.path.join(parent_path, args.processed_data_root)

    train_data = pd.read_csv(
        processed_data_path + data_name + ".train.rating", header=None, sep="\t"
    )
    test_data = pd.read_csv(
        processed_data_path + data_name + ".test.rating", header=None, sep="\t"
    )
    data = pd.concat([train_data, test_data])
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_context_data = torch.tensor(
        train_data.iloc[:, 3:].values.astype(float), dtype=torch.float
    )
    test_context_data = torch.tensor(
        test_data.iloc[:, 3:].values.astype(float), dtype=torch.float
    )

    train_context_data = TensorDataset(train_context_data)
    test_context_data = TensorDataset(test_context_data)

    # Dataloader
    train_loader = DataLoader(train_context_data, args.batch_size)
    test_loader = DataLoader(test_context_data, args.batch_size)

    # Model
    model = AutoEncoder(hidden_dims=args.layers, dropout=args.dropout, init_weights=args.init_weights).to(_device)

    # Initialize Optimizer and Loss function
    loss_fn = _loss_fn[args.loss]
    optimizers = _optimizers[args.optimizer](model.parameters(), lr=args.lr)
    losses = dict()
    min_loss = float("inf")

    # Init performace
    (rmse, mae) = evaluate_model(model, test_loader)
    print("Init: RMSE = %.4f, MAE = %.4f" % (rmse, mae))
    best_rmse, best_mae, best_iter = rmse, mae, -1
    rmse_dict = dict()
    mae_dict = dict()

    for epoch in range(args.epochs):
        print("Training epoch %d." % epoch)
        start_time = time()

        # Curriculum Learning
        train_epoch(
            optimizers, loss_fn, train_loader, model, losses, use_random_data=args.noise
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
            (rmse, mae) = evaluate_model(model, test_loader)
            test_time = ((time() - start_time) / 60) - train_time
            total_time = train_time + test_time

        rmse_dict[epoch] = rmse
        mae_dict[epoch] = mae

        print(
            f"[{epoch:d}] Elapsed time: {total_time:.2f}m - Train time: {train_time:.2f}m - Test time: {test_time:.2f}"
        )
        print(f"RMSE = {rmse:.4f}, MAE = {mae:.4f}, loss = {losses[epoch]:.4f}")

        # Save checkpoints
        chk_path = os.path.join(check_point_path, "epoch_{}.bin".format(epoch))
        chk_path_latest = os.path.join(check_point_path, "latest_epoch.bin")
        chk_path_best = os.path.join(check_point_path, "best_epoch.bin".format(epoch))
        model_params = {
            "hidden_dims": args.layers,
            "dropout": args.dropout,
            "activation": "ReLU",
        }
        # Save latest Model
        save_model_with_params(chk_path_latest, model, model_params)
        save_accuracy(
            check_point_path + "/latest_epoch",
            rmse=rmse,
            mae=mae,
            epoch=epoch,
            loss=losses[epoch],
        )

        # Save best Model based on RSME
        if rmse < best_rmse:
            best_rmse = rmse
            save_model_with_params(chk_path_best, model, model_params)
            save_accuracy(
                check_point_path + "/best_epoch",
                rmse=rmse,
                mae=mae,
                epoch=epoch,
                loss=losses[epoch],
            )

            # Plot and save losses every 23 hours 
            s2 = time()
            elapsed_time = s2 - s1
            if elapsed_time > (23 * 60 * 60):
                plot_and_save_dict(losses, check_point_path)
                plot_and_save_dict(rmse_dict, check_point_path, filename="rmse.png")
                plot_and_save_dict(mae_dict, check_point_path, filename="mae.png")

                save_model_specs(model, check_point_path)
                save_dict_to_file(args, check_point_path)
                save_dict_to_file(losses, check_point_path, filename="loses.txt")
                
                s1 = time()

    plot_and_save_dict(losses, check_point_path)
    plot_and_save_dict(rmse_dict, check_point_path, filename="rmse.png")
    plot_and_save_dict(mae_dict, check_point_path, filename="mae.png")

    save_model_specs(model, check_point_path)
    save_dict_to_file(args, check_point_path)
    save_dict_to_file(losses, check_point_path, filename="loses.txt")


if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_with_config(args, opts)

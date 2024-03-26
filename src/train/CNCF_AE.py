import argparse
from ast import Dict
from typing import Callable

import torch
from src.utils.model_stats.stats import save_model_with_params
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

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
        default="configs/CNCF/frappe-inter.yaml",
        help="Path to the config file.",
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
            latent = ae_model(context_input)['latent']
        
        output = rs_model(user_input, item_input, latent)
        loss = loss_fn(output, ratings)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    losses[len(losses)] = avg_loss  # Store the average loss for the epoch















model = save_model_with_params(chk_path)
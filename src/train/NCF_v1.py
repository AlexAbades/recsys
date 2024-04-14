import argparse
import errno
import os
from time import time
from typing import Callable, Dict

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer
from torch.nn import Module

from src.data.DataLoader import MovieLensDataset
from src.models.nfc.nfc import NFC
from src.utils.evaluation_metrics.evaluation import *
from src.utils.evaluation_metrics.evaluation import evaluate_model
from src.utils.model_stats.stats import (
    calculate_model_size,
    save_accuracy,
    save_checkpoint,
)
from src.utils.tools.tools import get_config, ROOT_PATH

# Debug = 1
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# TODO: Reciprocal Rank (get list of recomended items, see where is the first relevant item in the list, divide 1/position on the list), NCDG,
# TODO: Update Dataloader:
# - Add get Item
# - Update class so it only processed processed data, there is a class for preprocess data
# - Add option to generate negative samples ate the begining and not in every epoch.


def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="checkpoints/nfc",
        type=str,
        metavar="PATH",
        help="checkpoint directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nfc/frappe-1.yaml",
        help="Path to the config file.",
    )
    opts = parser.parse_args()
    return opts


_optimizers = {"adam": optim.Adam, "SGD": optim.SGD}
_loss_fn = {"BCE": nn.BCELoss()}

# Check for GPU
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        user_input = batch[0].to(_device)
        item_input = batch[1].to(_device)
        labels = batch[2].to(_device)
        labels = labels.view(-1, 1)

        output = model(user_input, item_input)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses[idx_loss] = loss


def train_with_config(args, opts):
    global _optimizers
    global _loss_fn
    global _device

    data_name, check_point_path = create_checkpoint_folder(args, opts)

    print(f"Running in device: {_device}")
    # Load Dataset
    print("Loading dataset ...")
    data = MovieLensDataset()
    processed_data_root = os.path.join(ROOT_PATH, args.processed_data_root, data_name)
    data.load_processed_data(processed_data_root)
    print("Generating Sparse Matrix...")
    train, testRatings, testNegatives = (
        data.trainMatrix,
        data.testRatings,
        data.testNegatives,
    )
    num_users, num_items = train.shape
    min_loss = 100000

    # Initialize Model
    model = NFC(
        num_users=num_users,
        num_items=num_items,
        mf_dim=args.num_factors,
        layers=args.layers,
    ).to(_device)
    # model_size_mb = calculate_model_size(model)

    # Init performance
    (hits, ndcgs) = evaluate_model(
        model, _device, testRatings, testNegatives, args.topK, args.evaluation_threads
    )

    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print("Init: HR = %.4f, NDCG = %.4f" % (hr, ndcg))

    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    # Define optimizer & loss
    loss_fn = _loss_fn[args.loss]
    optimizers = _optimizers[args.optimizer](model.parameters(), lr=args.lr)
    losses = dict()

    # Start epoch
    for epoch in range(args.epochs):
        print("Training epoch %d." % epoch)
        start_time = time()

        print("Generating Samples")
        user_input, item_input, labels = data.get_train_data()

        train_dataset = TensorDataset(user_input, item_input, labels)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        sample_time = (time() - start_time) / 60

        # Curriculum Learning
        train_epoch(optimizers, loss_fn, train_loader, model, losses)
        print(f"{epoch} and the loss: ", losses)
        # Calculate elapsed time for 1 train
        train_time = (time() - start_time) / 60 - sample_time

        # Evaluation
        if epoch % args.verbose == 0:
            (hits, ndcgs) = evaluate_model(
                model,
                _device,
                testRatings,
                testNegatives,
                args.topK,
                args.evaluation_threads,
            )
            hr, ndcg, loss = (
                np.array(hits).mean(),
                np.array(ndcgs).mean(),
                losses[epoch],
            )

            test_time = ((time() - start_time) / 60) - train_time - sample_time
            total_time = (time() - start_time) / 60

            print(
                "[%d] time %.2f m - HR = %.4f, NDCG = %.4f, loss = %.4f. Times: Data %.2f m - Train %.2f m - Test %.2f m"
                % (
                    epoch,
                    total_time,
                    hr,
                    ndcg,
                    loss,
                    sample_time,
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
                check_point_path + "/latest_epoch", hr=hr, ndcg=ndcg, epoch=epoch
            )

            # Save model every X epoch
            if (epoch + 1) % 20 == 0:
                save_checkpoint(
                    chk_path,
                    epoch,
                    args.lr,
                    optimizers,
                    model,
                    min_loss,
                )
                save_accuracy(check_point_path + f"/epoch_{epoch}", hr=hr, ndcg=ndcg)
            # Save best Model
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch

                save_checkpoint(
                    chk_path_best, epoch, args.lr, optimizers, model, min_loss
                )
                save_accuracy(
                    check_point_path + "/best_epoch", hr=hr, ndcg=ndcg, epoch=epoch
                )

def create_checkpoint_folder(args, opts):
    normalized_path = os.path.normpath(args.processed_data_root)
    data_name = os.path.basename(normalized_path)
    check_point_path = os.path.join(ROOT_PATH, opts.checkpoint, data_name, args.name)
    try:
        os.makedirs(check_point_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError(
                "Unable to create checkpoint directory:", check_point_path
            )
            
    return data_name,check_point_path


if __name__ == "__main__":

    opts = parse_args()
    args = get_config(opts.config)
    train_with_config(args, opts)

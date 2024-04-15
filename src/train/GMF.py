import argparse
import os
from collections import defaultdict
from time import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.cncf_collate_fn import ncf_negative_sampling
from src.data.nfc_dataset import NCFDataset
from src.models.GMF.gmf import GeneralMatrixFactorization
from src.utils.eval import getBinaryDCG, getHR, getRR
from src.utils.model_stats.stats import plot_and_save_losses, save_accuracy, save_checkpoint, save_dict_to_file, save_model_specs
from src.utils.tools.tools import (
    ROOT_PATH,
    create_checkpoint_folder,
    get_config,
    get_parent_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GMF model")
    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        default="configs/GMF/FRAPPE/frappe1.yaml",
        help="Configuration file for model specifications",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/GMF",
        help="Root folder of checkpoints",
    )
    return parser.parse_args()


_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_optimizers = {"adam": optim.Adam, "SGD": optim.SGD}
_loss_fn = {"BCE": nn.BCELoss(), "MSE": nn.MSELoss()}


def train_epoch(
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    model: nn.Module,
    losses: dict,
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

    for batch in train_loader:
        user_input = batch["user"].to(_device)
        item_input = batch["item"].to(_device)
        labels = batch["rating"].to(_device)
        labels = labels.view(-1, 1)

        output = model(user_input, item_input)
        loss = loss_fn(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    losses[idx_loss] = total_loss / num_batches

    return losses


def evaluate_model(model_pos: nn.Module, data_loader: DataLoader, topK: int):
    global _device

    model_pos.eval()

    all_users = []
    all_items = []
    all_predictions = []
    all_gtItems = []
    with torch.no_grad():
        for batch in data_loader:
            user_input = batch["user"].to(_device)
            item_input = batch["item"].to(_device)
            gtItems = batch["gtItem"]

            batch_predictions = model_pos(user_input, item_input)

            # TODO: Change the append for extened and flatten
            all_predictions.append(batch_predictions.cpu().numpy())
            all_users.append(user_input.cpu().numpy())
            all_items.append(item_input.cpu().numpy())
            all_gtItems.append(gtItems.numpy())

    # Concatenate all arrays into a single NumPy array
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
        topK_items = sorted(items_scores, key=lambda x: x[1], reverse=True)[:topK]
        gtItem = topK_items[0][2]
        topK_items = [item for item, _, _ in topK_items]

        # Evaluation
        hrs.append(getHR(topK_items, [gtItem]))
        rrs.append(getRR(topK_items, [gtItem]))
        ndcgs.append(getBinaryDCG(topK_items, [gtItem]))

    return np.mean(hrs), np.mean(rrs), np.mean(ndcgs)


def train_with_config(args, opts):
    global _optimizers, _loss_fn, _device

    # Folder structure checkpoint
    data_name, check_point_path = create_checkpoint_folder(args, opts)
    # log_path = os.path.join(ROOT_PATH, f"logs/logs_{args.foldername}")

    # Get parent folder
    parent_path = get_parent_path(ROOT_PATH)
    processed_data_path = os.path.join(parent_path, args.processed_data_root)

    print(f"Running in device: {_device}")

    train_data = NCFDataset(
        processed_data_path,
        split="train",
        n_items=args.num_items,
        num_negative_samples=5,
    )

    test_data = NCFDataset(
        processed_data_path,
        split="test",
        n_items=args.num_items,
        num_negative_samples=99,
    )

    # Dataloader
    train_loader = DataLoader(
        train_data, args.batch_size, collate_fn=ncf_negative_sampling
    )
    test_loader = DataLoader(
        test_data, args.batch_size, collate_fn=ncf_negative_sampling
    )

    # Number of users and items
    num_users = args.num_users
    num_items = args.num_items

    # Model
    model = GeneralMatrixFactorization(
        num_users=num_users, num_items=num_items, mf_dim=args.num_factors
    ).to(_device)

    # Initialize Optimizer & loss function
    optimizer = _optimizers[args.optimizer](model.parameters(), lr=args.lr)
    loss_fn = _loss_fn[args.loss]
    losses = {}
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
        train_epoch(optimizer, loss_fn, train_loader, model, losses)

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
            save_checkpoint(chk_path_latest, epoch, args.lr, optimizer, model, min_loss)
            save_accuracy(
                check_point_path + "/latest_epoch",
                hr=hr,
                mrr=mrr,
                ndcg=ndcg,
                epoch=epoch,
            )

            # Save best Model based on HR
            if hr > best_hr:
                best_hr = hr
                save_checkpoint(
                    chk_path_best, epoch, args.lr, optimizer, model, min_loss
                )
                save_accuracy(
                    check_point_path + "/best_epoch",
                    hr=hr,
                    mrr=mrr,
                    ndcg=ndcg,
                    epoch=epoch,
                )
    plot_and_save_losses(losses, check_point_path)
    save_model_specs(model, check_point_path)
    save_dict_to_file(args, check_point_path)
    save_dict_to_file(losses, check_point_path, filename="loses.txt")


if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_with_config(args, opts)

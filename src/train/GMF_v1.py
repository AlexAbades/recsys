import errno
import numpy as np
import yaml
from torch import nn, optim
import torch
from time import time
from src.data.cncf_interaction_dataset import CNCFDataset
from src.models.GMF.gmf_v1 import GMF
from torch.utils.data import DataLoader, TensorDataset
from src.utils.evaluation_metrics.evaluation import evaluate_model
import os

from src.utils.model_stats.stats import (
    calculate_model_size,
    save_accuracy,
    save_checkpoint,
)


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Load the configuration
config = load_config("./src/config/gmf/ml-1m-1.yaml")


# Accessing configuration data
optimizer = config["optimizer"]
learning_rate = config["learning_rate"]
batch_size = config["batch_size"]
num_epochs = config["epochs"]
num_negative_instances = config["num_negative_instances"]
num_factors = config["num_factors"]
data_path = "./src/data/processed/ml-1m/ml-1m"
optimizer = {"adam": optim.Adam, "SGD": optim.SGD}
loss_fn = {"BCE": nn.BCELoss()}
verbose = 1
topK = config["topK"]
evaluation_threads = config["evaluation_threads"]
check_point_path = "./src/checkpoints/gmf"

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(optimizer, loss_fn, device, model, start_time, sample_time, dataloader):
    for batch in dataloader:
        user_input = batch[0].to(device)
        item_input = batch[1].to(device)
        labels = batch[2].to(device)
        labels = labels.view(-1, 1)

            # Forward pass.
        output = model(user_input, item_input)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_time = (time() - start_time) / 60 - sample_time
    return train_time

if __name__ == "__main__":
    print(config)
    try:
        os.makedirs(check_point_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError(
                "Unable to create checkpoint directory:", check_point_path
            )

    print(f"Running in device: {device}")
    # Load Dataset
    print("Loading dataset ...")
    data = CNCFDataset()
    data.load_processed_data(data_path)
    print("Generating Sparse Matrix...")
    train, testRatings, testNegatives = (
        data.trainMatrix,
        data.testRatings,
        data.testNegatives,
    )
    num_users, num_items = train.shape
    min_loss = 100000

    # Initialize Model
    model = GMF(num_users=num_users, num_items=num_items, mf_dim=num_factors).to(device)
    model_size_mb = calculate_model_size(model)

    # Init performance
    (hits, ndcgs) = evaluate_model(
        model, device, testRatings, testNegatives, topK, evaluation_threads
    )
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print("Init: HR = %.4f, NDCG = %.4f" % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    # Define optimizer & loss
    loss_fn = loss_fn["BCE"]
    optimizer = optimizer["adam"](model.parameters(), lr=learning_rate)

    # Start epoch
    for epoch in range(num_epochs):
        start_time = time()

        model.train()

        print(f"Training epoch {epoch}. Generating N.S.")
        user_input, item_input, labels = data.get_train_data()
        sample_time = (time() - start_time) / 60

        dataset = TensorDataset(user_input, item_input, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        train_time = train_epoch(optimizer, loss_fn, device, model, start_time, sample_time, dataloader)

        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(
                model, device, testRatings, testNegatives, topK, evaluation_threads
            )
            hr, ndcg, loss = (
                np.array(hits).mean(),
                np.array(ndcgs).mean(),
                loss,
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
                chk_path_latest, epoch, learning_rate, optimizer, model, min_loss
            )
            save_accuracy(
                check_point_path + "/latest_epoch", hr=hr, ndcg=ndcg, epoch=epoch
            )

            # Save model every X epoch
            if (epoch + 1) % 20 == 0:
                save_checkpoint(
                    chk_path,
                    epoch,
                    learning_rate,
                    optimizer,
                    model,
                    min_loss,
                )
                save_accuracy(check_point_path + f"/epoch_{epoch}", hr=hr, ndcg=ndcg)
            # Save best Model
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch

                save_checkpoint(
                    chk_path_best, epoch, learning_rate, optimizer, model, min_loss
                )
                save_accuracy(
                    check_point_path + "/best_epoch", hr=hr, ndcg=ndcg, epoch=epoch
                )

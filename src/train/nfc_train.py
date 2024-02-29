import errno
from time import time
from src.models.nfc.nfc import NFC
from src.data.DataLoader import MovieLensDataset
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
from src.utils.evaluation_metrics.evaluation import evaluate_model
import numpy as np
import os
from src.utils.evaluation_metrics.evaluation import *
from src.utils.model_stats.stats import calculate_model_size, save_checkpoint

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# TODO: top popular items by users model to see that my model is performing better
# TODO: Write overleaf
# TODO: Table with perfomance on top popular, svd, NFC
# TODO: Reciprocal Rank (get list of recomended items, see where is the first relevant item in the list, divide 1/position on the list), NCDG,


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

print(os.getcwd())
# Load the configuration
config = load_config("./src/config/nfc/ml-1m-1.yaml")

# Accessing configuration data
layers = config["layers"]
learning_rate = config["learning_rate"]
optimizer = config["optimizer"]
batch_size = config["batch_size"]
num_epochs = config["epochs"]
num_negative_instances = config["num_negative_instances"]
dropout = config["dropout"]
num_factors = config["num_factors"]
data_path = "./src/data/processed/ml-1m/ml-1m"
optimizer = {"adam": optim.Adam, "SGD": optim.SGD}
loss_fn = {"BCE": nn.BCELoss()}
verbose = 1
topK = config["topK"]
evaluation_threads = config["evaluation_threads"]
check_point_path = "./src/checkpoints"

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    data = MovieLensDataset()
    data.load_processed_data(data_path)
    print(layers)
    

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
    data = MovieLensDataset()
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
    model = NFC(
        num_users=num_users, num_items=num_items, mf_dim=num_factors, layers=layers
    ).to(device)
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
        # Model to train
        model.train()

        print(f"Training epoch {epoch}.")
        print("Generating Negative Samples...")
        user_input, item_input, labels = data.get_train_data()
        # TODO: We can add get item, though if we have to genererate radom
        # negative samples each time we shoud include somthing to create each time a
        # datatensor is created.
        dataset = TensorDataset(user_input, item_input, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # calculate_memory_allocation()
        for batch in dataloader:
            user_input = batch[0].to(device)
            item_input = batch[1].to(device)
            labels = batch[2].to(device)
            labels = labels.view(-1, 1)

            # Forward pass.
            output = model(user_input, item_input)

            # Compute loss.
            loss = loss_fn(output, labels)

            # Clean up gradients from the model.
            optimizer.zero_grad()

            # Compute gradients based on the loss from the current batch (backpropagation).
            loss.backward()

            # Take one optimizer step using the gradients computed in the previous step.
            optimizer.step()

            # Calculate elapsed time for 1 train
            train_time = (time() - start_time) / 60
            break

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

            test_time = (time() - start_time) / 60

            print(
                "Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]"
                % (epoch, train_time, hr, ndcg, loss, test_time)
            )

            # Save checkpoints
            chk_path = os.path.join(check_point_path, "epoch_{}.bin".format(epoch))
            chk_path_latest = os.path.join(check_point_path, "latest_epoch.bin")
            chk_path_best = os.path.join(
                check_point_path, "best_epoch.bin".format(epoch)
            )

            save_checkpoint(
                chk_path_latest, epoch, learning_rate, optimizer, model, min_loss
            )
            # Save lastest model
            if (epoch + 1) % verbose == 0:
                save_checkpoint(
                    chk_path, epoch, learning_rate, optimizer, model, min_loss
                )

            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                save_checkpoint(
                    chk_path_best, epoch, learning_rate, optimizer, model, min_loss
                )
        break

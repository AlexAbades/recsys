import argparse
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
from src.utils.model_stats.stats import (
    calculate_model_size,
    save_accuracy,
    save_checkpoint,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


# TODO: top popular items by users model to see that my model is performing better
# TODO: Write overleaf
# TODO: Table with perfomance on top popular, svd, NFC
# TODO: Reciprocal Rank (get list of recomended items, see where is the first relevant item in the list, divide 1/position on the list), NCDG,


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="src/checkpoints/nfc/",
        type=str,
        metavar="PATH",
        help="checkpoint directory",
    )
    parser.add_argument(
        "--config", type=str, default="ml-1m-1.yaml", help="Path to the config file."
    )
    opts = parser.parse_args()
    return opts


# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    opts = parse_args()
    config_path = "src/config/nfc/" + opts.config
    # Load the configuration
    config = load_config(config_path)
    # Extract the base name (file name with extension)
    base_name = os.path.basename(opts.config)

    # Split the base name by '.' and discard the extension
    file_name_without_extension = os.path.splitext(base_name)[0]

    check_point_path = opts.checkpoint + file_name_without_extension
    print(check_point_path)

    try:
        os.makedirs(check_point_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError(
                "Unable to create checkpoint directory:", check_point_path
            )

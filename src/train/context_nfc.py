

import argparse
from torch import nn, optim
from torch.optim import Optimizer

def parse_args():
    parser = argparse.ArgumentParser(description="Run context Aware NCF.")
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="",
        type=str,
        metavar="PATH",
        help="checkpoint directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to the config file.",
    )
    opts = parser.parse_args()
    return opts

# TODO: Maybe extract as dictionary
_optimizers = {"adam": optim.Adam, "SGD": optim.SGD}
_loss_fn = {"BCE": nn.BCELoss(), "RSME": nn.MSELoss()}


if __name__ == "__main__":
  pass
from src.models.mlp.mlp import MLP
import os
from src.utils.tools.tools import get_config

PATH = "./configs/nfc/ml-1m-1.yaml"

if __name__ == "__main__":
    print(PATH)
    print(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
    print(os.path.abspath(__file__))
    print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    with open(PATH, "r") as stream:
        pass
    args = get_config(PATH)
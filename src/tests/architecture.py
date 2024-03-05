from src.models.mlp.mlp import MLP
import os
from src.utils.tools.tools import get_config

PATH = "configs/nfc/ml-1m-1.yaml"

if __name__ == "__main__":
    print(PATH)
    print(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
    print(os.path.abspath(__file__))
    ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    a = os.path.join(ROOT_PATH, PATH)
    print(a)
    args = get_config(a)
    print(args.name)


    # The in dimensions is the concatenation of user and item length or size, in this case both have same length
    # EMBEDDING_SIZE = 3
    # N_USERS = 5
    # # TODO: Understand Why N_ITEMS is the output size
    # N_ITEMS = 10
    # a = MLP(2 * EMBEDDING_SIZE, N_ITEMS, [6, 4, 2], dropout=0)
    # print(a)

from src.models.mlp.mlp import MLP
import os
from src.utils.tools.tools import get_config

from src.models.contextNFC.context_nfc import DeepNCF

if __name__ == "__main__":
    model = DeepNCF(num_users=6570, num_items=1012, num_context=22, mf_dim=8, layers=[42, 9, 4])
    print(model)
from src.data.BinaryClassifictionDataLoader import ContextDataLoaderBinaryClasifictaion
from src.data.Contextual import PreProcessDataNCFContextual
from src.models.mlp.mlp import MLP
import os
from src.utils.tools.tools import ROOT_PATH, get_config

from src.models.contextNFC.context_nfc import DeepNCF

if __name__ == "__main__":
    # datapath = "../data/raw/frappe/"
    batch_size = 100
    test_data = ContextDataLoaderBinaryClasifictaion(
        ROOT_PATH + "/data/processed/frappeCtxA",
        split="test",
        num_negative_samples=1000,
    )

    # df_frappe = PreProcessDataNCFContextual(datapath)

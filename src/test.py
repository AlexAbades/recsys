from src.data.Contextual import PreProcessDataNCFContextual
from src.models.mlp.mlp import MLP
import os
from src.utils.tools.tools import ROOT_PATH, get_config

from src.models.contextNFC.context_nfc import DeepNCF

if __name__ == "__main__":
    datapath = "../data/raw/frappe/"
    
    df_frappe = PreProcessDataNCFContextual(datapath)
    
    
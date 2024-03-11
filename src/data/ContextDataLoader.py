import pandas as pd
from pandas import DataFrame
import torch
from torch.utils.data import Dataset

# TODO: Define functions, loog __getitem__ to see the structure
class ContextDataLoader(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.data = self._load_data(data_path)

    def _load_data(self, data_path, sep) -> DataFrame:
        return pd.read_csv(data_path, sep=sep)

    def __getitem__(self, idx):
        return {
            "user": torch.tensor(self.data.loc[idx, 0], dtype=torch.long),
            "item": torch.tensor(self.data.loc[idx, 1], dtype=torch.long),
            "rating": torch.tensor(self.data.loc[idx, 2], dtype=torch.float),
            "context": torch.tensor(self.data.loc[idx, 3:], dtype=torch.long),
        }

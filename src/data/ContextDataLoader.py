import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset


class ContextDataLoader(Dataset):
    def __init__(self, data_path: str, sep: str = "\t") -> None:
        super().__init__()
        self.data = self._load_data(data_path, sep)

        # User, item, rating
        self.users = torch.tensor(self.data.iloc[:, 0].values, dtype=torch.long)
        self.num_users = len(self.users.unique())
        self.items = torch.tensor(self.data.iloc[:, 1].values, dtype=torch.long)
        self.num_items = len(self.items.unique())
        self.ratings = torch.tensor(self.data.iloc[:, 2].values, dtype=torch.float)
        # Context
        context_data = self.data.iloc[:, 3:].values.astype(float)
        self.contexts = torch.tensor(context_data, dtype=torch.float)  # Changed to float. The cnt (1st column) has to be float
        self.num_context = self.contexts.shape[1]

    def _load_data(self, data_path, sep) -> DataFrame:
        return pd.read_csv(data_path, sep=sep, header=None)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            "user": self.users[idx],
            "item": self.items[idx],
            "rating": self.ratings[idx],
            "context": self.contexts[idx],
        }

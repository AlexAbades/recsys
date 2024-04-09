from pandas import DataFrame
from torch.utils.data import Dataset
import pandas as pd


class CNCFDataLoader(Dataset):

    def __init__(
        self,
        folder_path,
        split: str,
        train_file: str = ".train.rating",
        test_file: str = ".test.rating",
        positive_file: str = ".positive_samples.pkl",
        num_negative_samples: int = 5,
        sep: str = "\t",
    ) -> None:
        super().__init__()

    def _load_data(self, data_path, sep) -> DataFrame:
        return pd.read_csv(data_path, sep=sep, header=None)

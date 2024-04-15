import os
import pickle
from random import sample
from typing import Dict

import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset


class CNCFDataset(Dataset):
    """
    A PyTorch Dataset class for loading CNCF interaction data.

    Parameters:
    - folder_path (str): The path to the folder containing the data files.
    - split (str): The split of the data to load (train or test).
    - train_file (str): The filename for the training data.
    - test_file (str): The filename for the test data.
    - positive_file (str): The filename for the positive samples data.
    - n_items (int): The total number of items in the dataset.
    - num_negative_samples (int): The number of negative samples to generate for each positive sample.
    - sep (str): The separator used in the data files.

    Attributes:
    - split (str): The split of the data being loaded.
    - train_file (str): The filename for the training data.
    - test_file (str): The filename for the test data.
    - positive_file (str): The filename for the positive samples data.
    - folder_path (str): The path to the folder containing the data files.
    - num_negative_samples (int): The number of negative samples to generate for each positive sample.
    - data_name (str): The name of the dataset.
    - train_path (str): The path to the training data file.
    - test_path (str): The path to the test data file.
    - positive_sample_path (str): The path to the positive samples data file.
    - items (set): A set of all item IDs in the dataset.
    - positive_samples (DataFrame): A DataFrame containing the positive samples for each user.
    - data (DataFrame): The loaded data as a DataFrame.

    Methods:
    - __getitem__(self, index): Get an item from the dataset.
    - __len__(self): Get the length of the dataset.
    - _load_data(self, split: str, sep: str) -> DataFrame: Load data from a file.
    - _check_split(self, split) -> str: Check if the split is valid.
    - _load_pkl(self, data_path: str) -> DataFrame: Load a pickled file.
    - _get_data_name(self, folder_path: str) -> str: Get the name of the dataset.
    - _get_global_paths(self): Get the paths to the data files.
    - _check_required_parameters(self, **kwargs): Check if required parameters are provided.
    """

    def __init__(
        self,
        folder_path: str = None,
        split: str = "train",
        train_file: str = ".train.rating",
        test_file: str = ".test.rating",
        positive_file: str = ".positive_samples.pkl",
        n_items: int = None,
        num_negative_samples: int = 4,
        sep: str = "\t",
    ) -> None:
        super().__init__()
        self._check_required_parameters(folder_path=folder_path, n_items=n_items)
        self.split = self._check_split(split)
        self.train_file = train_file
        self.test_file = test_file
        self.positive_file = positive_file
        self.folder_path = folder_path
        self.num_negative_samples = num_negative_samples
        self.data_name = self._get_data_name(folder_path)
        self.train_path, self.test_path, self.positive_sample_path = (
            self._get_global_paths()
        )
        self.items = set(range(n_items))
        self.positive_samples = self._load_pkl(self.positive_sample_path)
        self.data = self._load_data(self.split, sep)

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Parameters:
        - index (int): The index of the item to retrieve.

        Returns:
        dict: A dictionary containing the user, item, rating, and context information for the item.
        """
        if self.num_negative_samples == 0:
            return {
                "user": torch.tensor(self.data.iloc[index, 0], dtype=torch.long),
                "item": torch.tensor(self.data.iloc[index, 1], dtype=torch.long),
                "rating": torch.tensor(self.data.iloc[index, 2]),
                "context": torch.tensor(self.data.iloc[index, 3:]),
                "gtIem": torch.tensor(self.data.iloc[index, 1], dtype=torch.long),
            }

        user, item, rating, *context = self.data.iloc[index]
        positive_samples = self.positive_samples[user]
        negative_samples = list(self.items - positive_samples)
        negative_samples = sample(negative_samples, self.num_negative_samples)

        user = [user] * (self.num_negative_samples + 1)
        negative_ratings = [0] * self.num_negative_samples
        context = [context] * (self.num_negative_samples + 1)
        gtIems = [item] * (self.num_negative_samples + 1)

        return {
            "user": torch.tensor(user, dtype=torch.long),
            "item": torch.tensor([item] + negative_samples, dtype=torch.long),
            "rating": torch.tensor([rating] + negative_ratings),
            "context": torch.tensor(context),
            "gtItem": torch.tensor(gtIems),
        }

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
        int: The number of items in the dataset.
        """
        return len(self.data)

    def _load_data(self, split: str, sep: str) -> DataFrame:
        """
        Loads Data from the file. It can be train or test.

        Parameters:
        - split (str): The split to load. It can be train or test.
        - sep (str): The separator used in the file.

        Returns:
        DataFrame: The loaded data as a DataFrame.
        """
        if split == "train":
            return pd.read_csv(self.train_path, sep=sep, header=None)

        return pd.read_csv(self.test_path, sep=sep, header=None)

    def _check_split(self, split) -> str:
        """
        Check if the split is correct. It must be train or test.

        Parameters:
        - split (str): The split to check.

        Returns:
        str: The validated split.
        """
        if split not in ["train", "test"]:
            raise ValueError(f"Split must be train or test. Current {split}")
        return split

    def _load_pkl(self, data_path: str) -> Dict:
        """
        Load a pickled file from the given data path.Dictionary where keys are the
        user and values are the items the user has interacted with in a set.

        Parameters:
        - data_path (str): The path to the pickled file.

        Returns:
        DataFrame: The loaded pickled data as a DataFrame.
        """
        try:
            with open(data_path, "rb") as f:
                my_dict_loaded = pickle.load(f)
            return my_dict_loaded
        except Exception as e:
            print(f"Error loading file {data_path}")
            raise e

    def _get_data_name(self, folder_path: str) -> str:
        """
        Get the name of the dataset from the folder path.

        Parameters:
        - folder_path (str): The path to the folder containing the data files.

        Returns:
        str: The name of the dataset.
        """
        if folder_path.endswith("/"):
            folder_path = folder_path[:-1]
        return os.path.basename(folder_path)

    def _get_global_paths(self):
        """
        Get the paths to the data files.

        Returns:
        tuple: A tuple containing the paths to the training data, test data, and positive samples data files.
        """
        train_path = os.path.join(self.folder_path, self.data_name + self.train_file)
        test_path = os.path.join(self.folder_path, self.data_name + self.test_file)
        positive_sample_path = os.path.join(
            self.folder_path, self.data_name + self.positive_file
        )
        return train_path, test_path, positive_sample_path

    def _check_required_parameters(self, **kwargs):
        """
        Given a set of parameters, it throws an error if one of them is missing.

        Parameters:
        - kwargs: Keyword arguments representing parameters. Each parameter is a string.
        """
        # Check if any of the required parameters is None
        missing_params = [param for param, value in kwargs.items() if value is None]
        if missing_params:
            raise ValueError(
                f"Missing required parameters: {', '.join(missing_params)}"
            )

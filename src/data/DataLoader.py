import pandas as pd
from pandas import DataFrame
import numpy as np
import scipy.sparse as sp
from typing import Tuple, List
from torch.utils.data import Dataset
import torch
from torch import Tensor
from sklearn.model_selection import train_test_split

import pandas as pd

# TODO: Should create 2 main methods (aside from getitem)
# - For processing raw data, load dataset: Missing Create Negative Samples
# - Change file name to dataset, this is not a dataloader
# - Implement a __getitem__() method for the dataloader


class MovieLensDataset(Dataset):
    def __init__(
        self, test_size: float = 0.2, num_negatives: int = 4, sep: str = "::"
    ) -> None:
        """
        Custom Dataset that creates TrainMatrix, Test Negative and Test Data
        """
        self.testNegatives = None
        self.trainMatrix = None
        self.testRatings = None

        self.num_negatives = num_negatives
        self.test_size = test_size
        self.sep = sep

    def load_processed_data(self, path: str):
        """
        Function which loads the preprocessed data with files in form train.rating, test.rating & test.negative

        Args:
          - path: Relative path to the folder in where these files extensions are created
        """
        train_extension = ".train.rating"
        test_extension = ".test.rating"
        test_negative_extension = ".test.negative"
        self.testRatings = self._load_rating_file_as_list(path + test_extension)
        self.trainMatrix = self._load_rating_file_as_matrix(path + train_extension)
        self.testNegatives = self._load_negative_file(path + test_negative_extension)

    # TODO: Redundant, only makes sense if we want to retrun a list of rating, in that case we want it for test, which has the negative sampling
    def _load_rating_file_as_list(self, filepath: str) -> List[List[int]]:
        """
        Function that reads the .ratings which contains at least 2 Columns UserID & ItemID and

        Args:
          - filepath: The path to a file that contains the ratings

        Returns:
          - ratingList: List of pairs [User, Item]
        """
        ratingList = []
        with open(filepath, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                # Check If could be better a touple for inmutability
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def _load_rating_file_as_matrix(self, filepath: str) -> sp.dok_matrix:
        """
        Function that reads the .ratings which contains at least 3 Columns UserID & ItemID and Ratings

        Args:
          - filepath: The path to a file that contains the ratings

        Returns:
          - mat: Sparse matrix storing the positive instances for pairs (n_users, n_items).

        """
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filepath, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filepath, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if rating > 0:
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

    # TODO: We can modify the below function to create Negative and Test arrays
    def _load_negative_file(self, filepath) -> List[List[int]]:
        """
        Function that reads the .negative whch contains: first column of a Touple pairs indicating (UserID, ItemID)
        followed by K columns containg ItemIDs: ItemID_1, ItemID_2, ..., ItemID_k. Indicating negative samples for test.
        Each pair of UserID,ItemID corresponds to the ratings.test


        Args:
          - filepath: The path to a file that contains the ratings

        Returns:
          - negativeList: Retruns a List of List, First List indicates number od Users and the second the number of negative
            samples created per that pair
        """
        negativeList = []
        with open(filepath, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_raw_data(self):
        data = self._load_datasets(self.path)
        traindataset, testdataset = self._split_traintest(data, self.test_size)
        self.trainMatrix = self._create_sparse_matrix_from_dataset(traindataset)
        self.testRatings = self._create_testRatings_from_datset(testdataset)
        # TODO: Negative file: create a function

    def _load_datasets(self, ratings_path: str) -> DataFrame:
        return pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            names=["UserID", "MovieID", "Rating", "Timestamp"],
        )

    def _split_traintest(
        self, data: DataFrame, test_size: float
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Funtion that splits the dataser into train/test with a specific test size and a fixed seed for repr.

        Args:
          - data: raw dataframe with all user-item interactions
          - test_size: float indicating the % of split of the data
        """
        train, test = train_test_split(data, test_size=test_size, random_state=42)
        return train, test

    def _create_testRatings_from_datset(self, data: DataFrame) -> List[List[int]]:
        pairs_list = list(zip(data["UserID"], data["MovieID"]))
        # pairs_list_of_lists = [list(pair) for pair in pairs_list]
        return pairs_list

    def _create_sparse_matrix_from_dataset(
        self, train_dt: pd.DataFrame
    ) -> sp.dok_matrix:
        """
        Function that iterates through the dataset and generates a sparse matrix

        Args:
          - train_dt: trainig data to prepare to sparse matrix
        Returns:
          - sparse matrix storing the positive instances for pairs (n_users, n_items).
        """
        num_users = train_dt["UserID"].max()
        num_items = train_dt["MovieID"].max()
        # Construct matrix: Look out indexing
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        # Ensure that no 0 or lack of feedback is parsed as positve
        for _, row in train_dt.iterrows():
            user, item, rating = (
                int(row["UserID"]),
                int(row["MovieID"]),
                float(row["Rating"]),
            )
            if rating > 0:
                mat[user, item] = 1.0
        return mat

    def _create_negative_instances(
        self, train: sp.dok_matrix, num_negatives: int
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Create nº Negative instances per postive instance in the matrix.

        Args:
            - train: sparse matrix storing the positive instances for pairs (n_users, n_items).
            - num_negatives: The number of negative instances we want to generate per positive instacne.
        Reruns:
            - user_input: List of User IDs
            - item_input: List of Items IDs
            - labels: Binary List determine Positive or Negative Instance
        """
        user_input, item_input, labels = [], [], []
        num_items = train.shape[1]
        # Iterate over all non-zero elements in the sparse matrix
        # which are the positive instances
        train_coo = train.tocoo()  # Convert to COO format for easy iteration
        for u, i in zip(train_coo.row, train_coo.col):
            # positive instance
            user_input.append(u)
            item_input.append(i)
            labels.append(1)

            # negative instances
            for _ in range(num_negatives):
                j = np.random.randint(num_items)
                # Keep generating a new negative item until we find one that
                # the user has not interacted with
                while train[u, j] != 0:
                    j = np.random.randint(num_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)

        return user_input, item_input, labels

    def get_train_data(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Funtion that creates the necessary data to train the model.

        Returns:
          - user_input: Tensor of dimensions (nº of UserdId x nº of ratigs per user x nº of negative samples + 1) | Also:  (length of train dataset * 5)
          - item_input: Tensor of dimensions (nº of UserdId x nº of ratigs per user x nº of negative samples + 1) | Also:  (length of train dataset * 5)
          - labels: Tensor of dimensions (nº of UserdId x nº of ratigs per user x nº of negative samples + 1) | Also:  (length of train dataset * 5)
        """

        user_input, item_input, labels = self._create_negative_instances(
            self.trainMatrix, self.num_negatives
        )
        # Converting lists to PyTorch tensors
        user_input = torch.tensor(user_input, dtype=torch.long)
        item_input = torch.tensor(item_input, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float)

        return user_input, item_input, labels

    # TODO: Look the double tensors, decide if I want to store the user_input... as
    def __getitem__(self, idx):
        return {
            "user_input": torch.tensor(self.user_input[idx], dtype=torch.long),
            "item_input": torch.tensor(self.item_input[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }

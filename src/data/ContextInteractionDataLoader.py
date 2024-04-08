from random import sample, seed

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset
import os


class ContextInteractionDataLoader(Dataset):
    """
    Data Loader that given 2 Files. Inititalized a train or a test dateset.
    For the test set it can be specifyed the number of negative instances.  
    
    The structure of the data must be: 
    0 - User
    1 - Item 
    2 - Interaction
    3:end - Contextual
    """

    def __init__(
        self,
        folder_path,
        split: str,
        train_file: str = ".train.rating",
        test_file: str = ".test.rating",
        num_negative_samples: int = 5,
        sep: str = "\t",
    ) -> None:
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.folder_path = folder_path
        self.data_name = self._get_data_name(folder_path)
        self.sep = sep
        self.num_negative_samples = num_negative_samples
        self._initialize_atributes(split)

    def _load_data(self, data_path, sep) -> DataFrame:
        return pd.read_csv(data_path, sep=sep, header=None)

    def _get_data_name(self, folder_path: str) -> str:
        if folder_path.endswith("/"):
            folder_path = folder_path[:-1]
        return os.path.basename(folder_path)

    def _initialize_atributes(self, split: str):
        if split not in ["train", "test"]:
            raise AttributeError("split should be test ot train")

        # Initialize train set
        train_path = os.path.join(self.folder_path, self.data_name + self.train_file)
        self.train_data = self._load_data(train_path, self.sep)
        self.train_users = self.train_data.iloc[:, 0].values
        self.train_items = self.train_data.iloc[:, 1].values
        self.unique_items = set(self.train_items)

        # Initialize interacted an non interacted items per user
        self.items_per_user_train = self.train_data.groupby(0)[1].apply(set).to_dict()
        self.non_interacted_items_per_user = {
            user: list(self.unique_items - items_set)
            for user, items_set in self.items_per_user_train.items()
        }

        if split == "train":

            # User, item, rating Train
            self.train_ratings = self.train_data.iloc[:, 2].values
            # TODO: self.trainContext not used
            # Context Features Train
            self.train_context_data = self.train_data.iloc[:, 3:]
            self.trainContext = torch.tensor(
                self.train_context_data.values.astype(float),
                dtype=torch.float,  # should we can change to float? context comes from one-hot encoding
            )

            self.num_context = self.trainContext.shape[1]
            self.num_users = len(np.unique(self.train_users))
            self.num_items = len(np.unique(self.train_items))

            self.unique_users = set(self.train_users)

            self.create_train_data()

        else:
            # Load Test Data
            train_path = os.path.join(self.folder_path, self.data_name + self.test_file)
            self.test_data = self._load_data(train_path, self.sep)

            # User, item, rating Train
            self.test_users = self.test_data.iloc[:, 0].values
            self.test_items = self.test_data.iloc[:, 1].values
            self.test_ratings = self.test_data.iloc[:, 2].values

            # Context Features Test
            self.test_context_data = self.test_data.iloc[:, 3:]
            self.test_context = torch.tensor(
                self.test_context_data.values.astype(float), dtype=torch.float
            )
            self.num_context = self.test_context.shape[1]

            self.create_test_data()

    def create_test_data(self):

        # Initialize new sets
        extended_users, extended_items, extended_rating, extended_context, gtItems = (
            [],
            [],
            [],
            [],
            [],
        )
        for user, item, context in zip(
            self.test_users, self.test_items, self.test_context_data.values
        ):
            # Add the current item and a positive rating
            extended_users.append(user)
            extended_items.append(item)
            extended_context.append(context)
            extended_rating.append(1)

            # Remove the test item if it is in the non_interacted Items
            non_iteracted_items_user = self.non_interacted_items_per_user[int(user)]
            try:
                non_iteracted_items_user.remove(item)
            except ValueError:
                pass
            if self.num_negative_samples < len(non_iteracted_items_user):
                negative_samples = sample(
                    non_iteracted_items_user,
                    k=self.num_negative_samples,
                )
            else:
                negative_samples = non_iteracted_items_user

            # Extend lists with negative samples and corresponding ratings
            extended_users.extend([user] * self.num_negative_samples)
            extended_items.extend(negative_samples)
            gtItems.extend([item] * (self.num_negative_samples + 1))
            extended_context.extend([context] * self.num_negative_samples)
            extended_rating.extend([0] * self.num_negative_samples)

        self.extended_users = torch.tensor(extended_users, dtype=torch.long)
        self.extended_items = torch.tensor(extended_items, dtype=torch.long)
        self.extended_context = torch.tensor(
            np.array(extended_context), dtype=torch.float
        )
        self.extended_ratings = torch.tensor(extended_rating, dtype=torch.float)
        self.gtItems = torch.tensor(gtItems, dtype=torch.long)
        self.length_extended_data = len(extended_users)

    def create_train_data(self):
        """
        This method should be called once every epoch to ensure randomness among the negative samples.
        """
        # Create Context
        context_repeated_index = np.repeat(
            self.train_context_data.index, (self.num_negative_samples + 1)
        )
        expanded_context = self.train_context_data.loc[
            context_repeated_index
        ].reset_index(drop=True)
        self.extended_context = torch.tensor(
            expanded_context.values.astype(float), dtype=torch.float
        )

        # Initialize new sets
        extended_users, extended_items, extended_rating = [], [], []

        for user, item in zip(self.train_users, self.train_items):
            # Add the current item and a positive rating
            extended_users.append(user)
            extended_items.append(item)
            extended_rating.append(1)

            # Sample negative items
            negative_samples = sample(
                self.non_interacted_items_per_user[int(user)],
                k=self.num_negative_samples,
            )

            # Extend lists with negative samples and corresponding ratings
            extended_users.extend([user] * self.num_negative_samples)
            extended_items.extend(negative_samples)
            extended_rating.extend([0] * self.num_negative_samples)

        self.extended_users = torch.tensor(extended_users, dtype=torch.long)
        self.extended_items = torch.tensor(extended_items, dtype=torch.long)
        self.gtItems = torch.tensor([], dtype=torch.long)
        self.extended_ratings = torch.tensor(extended_rating, dtype=torch.float)
        self.length_extended_data = len(extended_users)

    def __len__(self):
        return self.length_extended_data

    def __getitem__(self, idx):

        return {
            "user": self.extended_users[idx],
            "item": self.extended_items[idx],
            "context": self.extended_context[idx],
            "rating": self.extended_ratings[idx],
            "gtItem": self.gtItems[idx] if self.gtItems.numel() else 0,
        }

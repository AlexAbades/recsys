from random import sample
from torch.utils.data import Dataset
import pandas as pd
from pandas import DataFrame
import torch
from random import sample, seed


class ContextDataLoader(Dataset):
    def __init__(
        self, data_path, num_negative_samples: int = 5, sep: str = "\t"
    ) -> None:
        super().__init__()
        self.data = self._load_data(data_path, sep)
        # User, item, rating
        self.users = torch.tensor(self.data.iloc[:, 0].values, dtype=torch.long)
        self.num_users = len(self.users.unique())
        self.items = torch.tensor(self.data.iloc[:, 1].values, dtype=torch.long)
        self.num_items = len(self.items.unique())
        self.ratings = torch.tensor(self.data.iloc[:, 2].values, dtype=torch.float)
        self.unique_items = set(self.users.unique())
        self.unique_users = set(self.items.unique())
        self.num_negative_samples = num_negative_samples
        seed(42)

        # Creating a dictionary to keep track of items each user has interacted with in train_data
        self.items_per_user_train = self.data.groupby(0)[1].apply(set).to_dict()

        # Convert context features, ensuring they're numeric (float)
        context_data = self.data.iloc[:, 3:].values.astype(float)
        self.contexts = torch.tensor(context_data, dtype=torch.long)
        self.num_context = self.contexts.shape[1]

    def _load_data(self, data_path, sep) -> DataFrame:
        return pd.read_csv(data_path, sep=sep, header=None)

    def __len__(self):
        return len(self.ratings)

    def _create_one_negative_sample(self, user):
        items_interacted = self.items_per_user_train[user]
        non_interacted_items = list(self.unique_items - items_interacted)
        negative_samples = sample(non_interacted_items, n=1)
        # TODO: We can create the list of non interated items every time a new user comes
        return torch.tensor(negative_samples)

    def __getitem__(self, idx):

        # Determine if this request is for a positive or negative sample
        if idx % (self.num_negatives + 1) == 0:
            # Positive sample
            user = self.users[idx]
            item = self.items[idx]
            context = self.contexts[idx]
            rating = 1

            items_interacted = self.items_per_user_train[user]
            non_interacted_items = list(self.unique_items - items_interacted)
        else:
            user = self.users[idx]
            item = torch.tensor(sample(non_interacted_items, n=1))
            context = self.contexts[idx]
            rating = 0

        return {
            "user": user,
            "item": item,
            "rating": context,
            "context": rating,
        }

    #   user = self.users[idx]
    #   item = self.items[idx]
    #   context = self.contexts[idx]
    #   rating = self.ratings[idx]
    #   negative_samples = self._create_negative_sample(user)

    #   sample_users = user.repeat(self.num_negative_samples+1)
    #   item_samples = torch.cat((item, negative_samples))
    #   context_samples = context.repeat(self.num_negative_samples+1)
    #   rating_samples = rating.repeat(self.num_negative_samples+1)

    #   return {
    #       "user": sample_users,
    #       "item": item_samples,
    #       "rating": context_samples,
    #       "context": rating_samples,
    #   }

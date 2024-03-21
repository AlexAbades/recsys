from random import sample
from torch.utils.data import Dataset
import pandas as pd
from pandas import DataFrame
import torch
from random import sample, seed


# TODO: If we want to test the model with all items, we should exclude the train items. This gives an
# non equal number of items per user in the test.

class ContextDataLoaderBinaryClasifictaionOneWorker(Dataset):
    """
    Atention, for using this class, the number of workers must be 0 and the shuffle set to Fasle
    """

    def __init__(
        self, data_path, num_negative_samples: int = 5, sep: str = "\t"
    ) -> None:
        super().__init__()
        self.data = self._load_data(data_path, sep)

        # User, item, rating
        self.trainUsers = torch.tensor(self.data.iloc[:, 0].values, dtype=torch.long)
        self.trainItems = torch.tensor(self.data.iloc[:, 1].values, dtype=torch.long)
        self.trainRatings = torch.tensor(self.data.iloc[:, 2].values, dtype=torch.float)

        self.num_users = len(self.trainUsers.unique())
        self.num_items = len(self.trainItems.unique())

        self.unique_users = set(self.trainUsers.unique())
        self.unique_items = set(self.trainItems.unique())

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
        return len(self.trainRatings) * (1 + self.num_negative_samples)

    def __getitem__(self, idx):

        # Decide if this is a positive or a negative sample
        is_positive_sample = idx % (self.num_negative_samples + 1) == 0

        # Determine if this request is for a positive or negative sample
        if is_positive_sample:
            # Get Real User and Real Context
            self.user_batch = self.trainUsers[idx]
            self.context_batch = self.contexts[idx]

            # Get the items the user has interacted with
            user_interacted_items = self.items_per_user_train.get(
                self.user_batch.item(), set()
            )
            self.user_non_interacted_items_batch = list(
                self.unique_items - user_interacted_items
            )

            s = set(range(10))
            sampled_item = sample(list(s), 1)[0]
            s.remove(sampled_item)

            # Positive sample
            item = self.trainItems[idx]
            rating = 1

        else:
            # Negative
            item = torch.tensor(
                sample(self.user_non_interacted_items_batch, k=1), dtype=torch.long
            )
            self.user_non_interacted_items_batch.remove(item[0])
            rating = 0

        return {
            "user": self.user_batch,
            "item": item,
            "context": self.context_batch,
            "rating": torch.tensor(rating, dtype=torch.float),
        }

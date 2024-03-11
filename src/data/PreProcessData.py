import errno
import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd


class PreProcessData:
    def __init__(
        self,
        data_path: str = None,
        user_column: str = "user",
        item_column: str = "item",
        interaction_column: str = "cnt",
        sep: str = "\t",
    ):
        """
        Initialize a class to pre-process data and convert it into

        Args:
            - dataPath: path to the folder of the raw data
            - user_column: Column which holds the users Ids
            - item_column: Column which holds the item Ids
            - interaction_column: Column that holds the interaction of pair u-i. Must be a number

        """
        self.data_path = data_path
        self.user_column = user_column
        self.item_column = item_column
        self.interaction_column = interaction_column
        self.sep = sep
        self.rawData = self._load_data(data_path)
        self.ratings = self._binarize_data()

    def _load_data(self, data_path):
        return pd.read_csv(filepath_or_buffer=data_path, sep=self.sep)

    def split_traintest(self):
        """
        Funtion that splits the dataser into train/test. It ensures tha no users with just one interaction
        end up in the test set.

        """
        frequency_interaction = self.ratings.groupby(self.user_column)[
            [self.item_column, self.interaction_column]
        ].count()
        users_one_interaction = frequency_interaction[
            frequency_interaction[self.interaction_column] == 1
        ].index
        list_of_users = self.ratings[self.user_column].unique()
        users_more_one_interaction = list(
            set(list_of_users) - set(users_one_interaction)
        )
        idx = []
        for i in users_more_one_interaction:
            element = self.ratings[self.ratings["user"] == i].sample(n=1)
            idx.append(element.index[0])

        test_idx = np.isin(self.ratings.index, np.array(idx))

        self.train_ratings, self.test_ratings = (
            self.ratings[~test_idx],
            self.ratings[test_idx],
        )

    def _binarize_data(self):
        """
        Function that binarizes the interaction column of the data. It can be ratigs, downloads, impresions...
        The column to binarize must be a float or integer column.
        """
        if not self.interaction_column:
            self.ratings = self.rawData[
                [self.user_column, self.item_column, self.interaction_column]
            ].copy()
            self.ratings[self.interaction_column] = self.ratings[
                self.interaction_column
            ].apply(lambda x: 1 if x > 0 else 0)

    def negative_samples(self, K: int = 99):
        """
        Function that generates K negative sample for the selected test set.
        It generates a negative array with the following structure:
        [[(userID_1,itemID_0)\t negativeItemID1\t negativeItemID2 ... negativeItemIDK], ...]

        Args:
            - K: the numbernegative samples for the test
        """
        set_of_items = set(self.rawData[self.item_column].unique())
        seen_items_by_user = (
            self.rawData.groupby(self.user_column)[self.item_column]
            .apply(set)
            .to_dict()
        )
        negative_samples = []
        for row in self.test_ratings.iterrows():

            user = row[1][self.user_column]
            item = row[1][self.item_column]
            unseen_items = list(set_of_items - seen_items_by_user.get(user, set()))
            random_elements = random.sample(unseen_items, K)

            # Ensure there are at least K unseen items
            if len(unseen_items) < K:
                raise ValueError(
                    f"User {user} does not have enough unseen items to select {K} random elements."
                )

            random_elements = random.sample(unseen_items, K)
            negative_samples.append([(user, item)] + random_elements)
        self.test_negative = pd.DataFrame(negative_samples)

    def save_data(self, folder_name):
        """
        Saves the Class atributes train-ratings,
        """
        current_file_path = os.path.abspath(__file__)
        data_folder_path = os.path.dirname(current_file_path)
        processed_data_path = os.path.join(data_folder_path, "processed", folder_name)
        folder_path = os.path.join(processed_data_path, folder_name)
        print(folder_path)
        try:
            os.makedirs(processed_data_path)

        except OSError as e:
            if e.errno != errno.EEXIST:
                raise RuntimeError(
                    "Unable to create checkpoint directory:", processed_data_path
                )
        except Exception as e:
            raise RuntimeError(f"An error occurred: {e}")
        self.train_ratings.to_csv(
            folder_path + ".train.rating",
            index=False,
            sep=self.sep,
            header=False,
        )
        self.test_ratings.to_csv(
            folder_path + ".test.rating",
            index=False,
            sep=self.sep,
            header=False,
        )
        self.test_negative.to_csv(
            folder_path + ".test.negative",
            index=False,
            sep=self.sep,
            header=False,
        )

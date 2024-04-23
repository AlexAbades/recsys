import math
import os
from typing import List, Tuple

import pandas as pd
from pandas import DataFrame


class PopularItem:
    def __init__(
        self,
        path: str,
        data_name:str,
        K: int,
        user_column: str = None,
        item_column: str = None,
        rating_column: str = None,
    ) -> None:
        """
        Constructor, it creates automatically a list of Popular Items of length K given
        the path of the .train.rating and test.rating

        Args:
            - path: Path to the folder with the train test files.
            - K: Number of elements of the popular item list
            - user_column: Name of the column in the train and test files that represents the user.
            - item_column: Name of the column in the train and test files that represents the item.
            - rating_column: Name of the column in the train and test files that represents the rating.
        """

        self.path = path
        self.K = K
        self.user_column = user_column
        self.item_column = item_column
        self.rating_column = rating_column

        # Load data as soon as an instance is created
        self.dt_train, self.dt_test = self._load_data(data_name)
        self.popular_items = self._calculate_K_popular_items(K)
        self.hit_ratio = self.calculate_hit_ratio(self.popular_items)
        self.ndcg_ratio = self.calculate_ndcg_ratio_binary(self.popular_items)
        self.mrr = self.calculate_mrr(self.popular_items)

    def _load_data(self, data_name) -> Tuple[DataFrame, DataFrame]:
        """
        Load the train and test data from the specified path.

        Returns:
            Tuple: A tuple containing the train and test data as pandas DataFrames.
        """

        train_path = os.path.join(self.path, f"{data_name}.train.rating")
        test_path = os.path.join(self.path, f"{data_name}.test.rating")

        dt_train = pd.read_csv(
            train_path,
            sep="\t",
            names=[self.user_column, self.item_column, self.rating_column],
        )
        dt_test = pd.read_csv(
            test_path,
            sep="\t",
            names=[self.user_column, self.item_column, self.rating_column],
        )

        return dt_train, dt_test

    def _calculate_K_popular_items(self, K):
        """
        Calculate the K most popular items based on the train data.

        Parameters:
            K: Number of popular items to calculate.

        Returns:
            List: A list of K most popular items.
        """

        popular_items = list(
            self.dt_train[self.dt_train[self.rating_column] != 0]
            .groupby(self.item_column)
            .count()[self.rating_column]
            .sort_values(ascending=False)
            .head(K)
            .index
        )
        return popular_items

    def calculate_hit_ratio(self, popular_items):
        """
        Calculate the hit ratio based on the test data.

        Parameters:
            popular_items: List of popular items.

        Returns:
            float: The hit ratio.
        """

        average_ht = (
            self.dt_test[self.item_column]
            .apply(lambda x: 1 if x in popular_items else 0)
            .mean()
        )
        return average_ht

    def getDCG(self, ranklist: list, gtItem: int) -> float:
        """
        Calculate the Discounted Cumulative Gain (DCG) based on the position of the ground truth item in the ranked list.

        Parameters:
            ranklist: The ranked list of items.
            gtItem: The ground truth item.

        Returns:
            float: The DCG value.
        """

        for i, item in enumerate(ranklist):
            if item == gtItem:
                return math.log(2) / math.log(i + 2)  # Using log base 2
        return 0

    def calculate_ndcg_ratio_binary(self, popular_items: list) -> float:
        """
        Calculate the Normalized Discounted Cumulative Gain (NDCG) ratio based on binary relevance.

        Parameters:
            popular_items: List of popular items.

        Returns:
            float: The NDCG ratio.
        """

        ncdgs = []
        idcg = 1
        for _, row in self.dt_test.iterrows():
            dcg = self.getDCG(popular_items, row[self.item_column])
            ncdg = dcg / idcg
            ncdgs.append(ncdg)
        average_ncdg = sum(ncdgs) / len(ncdgs)
        return average_ncdg

    def get_rr(self, ranklist: list, gtItem: int) -> float | int:
        """
        Calculate the Reciprocal Rank (RR) of the ground truth item in the ranked list.

        Parameters:
            ranklist: The ranked list of items.
            gtItem: The ground truth item.

        Returns:
            float | int: The RR value.
        """

        for i, item in enumerate(ranklist):
            if item == gtItem:
                return 1 / (i + 1)
        return 0

    def calculate_mrr(self, popular_items: list) -> float:
        """
        Calculate the Mean Reciprocal Rank (MRR) based on the test data.

        Parameters:
            popular_items: List of popular items.

        Returns:
            float: The MRR value.
        """

        rrs = []
        for _, row in self.dt_test.iterrows():
            rr = self.get_rr(popular_items, row[self.item_column])
            rrs.append(rr)
        average_rr = sum(rrs) / len(rrs)
        return average_rr

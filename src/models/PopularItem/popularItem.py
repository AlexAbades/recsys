import math
from typing import List

import pandas as pd


# TODO: Document Functions + class
# TODO: Extract the evaluation functions in file (check evaluation on model)
class PopularItem:
    def __init__(self, path: str, K: int) -> None:
        """
        Constructor, it creates automatically a list of Popular Items of length K given
        the path of the .train.rating and test.rating

        Args:
            - path:path to the folder with the train test files.
            - K: Number of elements of the popular item list
        """
        # TODO: Just for preprocessed list
        self.path = path
        self.K = K
        # Load data as soon as an instance is created
        self.dt_train, self.dt_test = self._load_data()
        self.popular_items = self._calculate_K_popular_items(K)

    def _load_data(self):
        # Private method to load data
        train_extension = ".train.rating"
        test_extension = ".test.rating"

        dt_train = pd.read_csv(
            f"{self.path}{train_extension}",
            sep="\t",
            names=["UserID", "MovieID", "Rating", "Timestamp"],
        )
        dt_test = pd.read_csv(
            f"{self.path}{test_extension}",
            sep="\t",
            names=["UserID", "MovieID", "Rating", "Timestamp"],
        )

        return dt_train, dt_test

    def _calculate_K_popular_items(self, K):
        popular_items = list(
            self.dt_train[self.dt_train["Rating"] != 0]
            .groupby("MovieID")
            .count()["Rating"]
            .sort_values(ascending=False)
            .head(K)
            .index
        )
        return popular_items

    def calculate_hit_ratio(self, K):

        # Calculate hit ratio based on test data
        average_ht = (
            self.dt_test["MovieID"]
            .apply(lambda x: 1 if x in self.popular_items else 0)
            .mean()
        )
        return average_ht

    def getDCG(ranklist, gtItem):
        """Calculate DCG based on the position of the ground truth item in the ranked list."""
        for i, item in enumerate(ranklist):
            if item == gtItem:
                return math.log(2) / math.log(i + 2)  # Using log base 2
        return 0

    def calculate_ndcg_ratio_binary(self, K):
        ncdgs = []
        idcg = 1
        for _, row in self.dt_test.iterrows():
            dcg = self.getDCG(self.popular_items[:K], row["MovieID"])
            ncdg = dcg / idcg
            ncdgs.append(ncdg)
        average_ncdg = sum(ncdgs) / len(ncdgs)
        return average_ncdg


def hitRatio(ranklist: List, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist: List, gtItem):
    """
    Args:
      - ranklist (List): The list of items, ranked according to some criteria.
      - gtItem: The ground truth item for which the NDCG score is to be calculated.
    Returns:
      - A floating-point number representing the NDCG score
    """
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0.0

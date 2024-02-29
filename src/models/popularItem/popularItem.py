import pandas as pd
import math


# TODO: Document Functions + class
# TODO: Extract the evaluation functions in file (check evaluation on model)
class PopularItem:
    def __init__(self, path: str) -> None:
        # TODO: Just for preprocessed list+
        self.path = path
        # Load data as soon as an instance is created
        self.dt_train, self.dt_test = self._load_data()

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

    def calculate_hit_ratio(self, K):
        # Calculate popular items based on training data
        self.popular_items = list(
            self.dt_train[self.dt_train["Rating"] != 0]
            .groupby("MovieID")
            .count()["Rating"]
            .sort_values(ascending=False)
            .head(K)
            .index
        )
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

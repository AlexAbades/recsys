import errno
import os
from typing import List
from pandas import DataFrame
import pandas as pd
import numpy as np


class PreProcessDataNCFContextual:
    def __init__(
        self,
        path: str,
        data_file: str = "frappe.csv",
        meta_file: str = "meta.csv",
        user_column: str = "user",
        item_column: str = "item",
        ctx_categorical_columns: List[str] = [
            "daytime",
            "weather",
            "isweekend",
            "homework",
        ],
        ctx_numerical_columns: List[str] = ["cnt"],
        ratings_colum: str = "rating",
        sep: str = "\t",
    ) -> None:
        self.path = path
        self.ratings_column = ratings_colum
        self.user_column = user_column
        self.item_column = item_column
        self.columns = self._clean_columns(
            user_column,
            item_column,
            ratings_colum,
            ctx_categorical_columns,
            ctx_numerical_columns,
        )
        self.sep = sep
        self.rawData = self._load_data(path + data_file, sep)
        self.rawMeta = self._load_data(path + meta_file, sep)
        self.data = self.create_data(
            ratings_colum, item_column, ctx_categorical_columns, ctx_numerical_columns
        )
        self._initialize_train_test(self.data)

    def _load_data(self, data_path, sep) -> DataFrame:
        return pd.read_csv(data_path, sep=sep)

    def _merge_data(self, data, metadata, item_column) -> DataFrame:
        dataframe = data.merge(metadata, on=item_column)
        return dataframe

    def _clean_columns(self, *args) -> List[str]:
        clean_columns = []
        for i in args:
            if isinstance(i, str):
                clean_columns.append(i)
                continue
            clean_columns.extend(i)
        return clean_columns

    def create_data(
        self,
        ratings_colum: str,
        item_column: str,
        ctx_categorical_columns: List[str],
        ctx_numerical_columns: List[str],
    ) -> DataFrame:
        data = self._merge_data(self.rawData, self.rawMeta, item_column)[self.columns]
        data = self.clear_ratings(data, ratings_colum)
        data = self.log_numerical(data, ctx_numerical_columns)
        data = data[self.columns]
        data = self.binarize_nominal_features(data, ctx_categorical_columns)
        return data

    def clear_ratings(self, df: DataFrame, rating_column: str) -> DataFrame:
        df[rating_column] = pd.to_numeric(df[rating_column], errors="coerce")
        df = df.dropna(subset=[rating_column])
        return df

    def log_numerical(self, df: DataFrame, numerical_features: List[str]) -> DataFrame:
        df_tmp = df.copy()
        for feature in numerical_features:
            df_tmp[feature] = df_tmp[feature].apply(
                lambda x: np.log10(x) if x != 0 else np.log10(x + 1)
            )
        return df_tmp

    def binarize_nominal_features(
        self, df: DataFrame, contextual_features: List[str]
    ) -> DataFrame:
        df_bin = pd.get_dummies(df, columns=contextual_features, dtype=int)
        return df_bin

    def _initialize_train_test(self, data: DataFrame):
        """
        Funtion that splits the dataser into train/test. It ensures that no users with just one interaction
        end up in the test set.

        """
        frequency_interaction = data.groupby(self.user_column)[
            [self.item_column, self.ratings_column]
        ].count()
        users_one_interaction = frequency_interaction[
            frequency_interaction[self.ratings_column] == 1
        ].index
        list_of_users = self.data[self.user_column].unique()
        users_more_one_interaction = list(
            set(list_of_users) - set(users_one_interaction)
        )
        idx = []
        for i in users_more_one_interaction:
            element = self.data[self.data["user"] == i].sample(n=1)
            idx.append(element.index[0])

        test_idx = np.isin(self.data.index, np.array(idx))

        self.train_ratings, self.test_ratings = (
            self.data[~test_idx],
            self.data[test_idx],
        )

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

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

        self.columns = self._clean_columns(
            user_column,
            item_column,
            ctx_categorical_columns,
            ctx_numerical_columns,
            ratings_colum,
        )
        self.rawData = self._load_data(path + data_file, sep)
        self.rawMeta = self._load_data(path + meta_file, sep)
        self.data = self.create_data(
            ratings_colum, item_column, ctx_categorical_columns, ctx_numerical_columns
        )

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

import errno
import os
import pandas as pd
from typing import Tuple
from pandas import DataFrame
from sklearn.model_selection import train_test_split


class PreProcessData:
    def __init__(self, path=None):
        self.path = path
        # self.dataframe = self._load_data(path)

    def _load_data(self, path):
        return pd.read_csv(self.path, sep="\t")

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

    def _binarize_data(data: DataFrame, column_to_binarize: str = None):
        df_tmp = data[["user", "item", column_to_binarize]].copy()
        df_tmp[column_to_binarize] = df_tmp[column_to_binarize].apply(
            lambda x: 1 if x > 0 else 0
        )
        df_tmp.to_csv(".train.ratings", index=True, sep="\t", header=False)

        return

    def _save_data(folder_name):
        current_file_path = os.path.abspath(__file__)
        data_folder_path = os.path.dirname(current_file_path)
        processed_data_path = os.path.join(data_folder_path, "processed", folder_name)
        try:
            os.makedirs(processed_data_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise RuntimeError(
                    "Unable to create checkpoint directory:", processed_data_path
                )

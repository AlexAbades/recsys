import errno
import os
import textwrap
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# TODO: We need a function that binarized the data?


class PreProcessDataNCFContextual:
    """
    Preprocess script.
    Given a data set with categorical and numercial features the script performs filtering, 
    data transformation, normalization, and one-hot encoding on the dataset.

    TODO: We have to modify the function so it also accepts one file with all features.
    
    """
    def __init__(
        self,
        path: str,
        data_file: str = "frappe.csv",
        meta_file: str = "meta.csv",
        user_column: str = "user",
        item_column: str = "item",
        ratings_colum: str = "rating",
        ctx_categorical_columns: List[str] = [
            "daytime",
            "weather",
            "isweekend",
            "homework",
        ],
        ctx_numerical_columns: List[str] = ["cnt"],
        columns_to_normalize: List[str] = ["cnt"],
        min_interactions: int = 5,
        min_samples_per_user_test_set: int = 1,
        binary_classification: bool = False,
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
        self.min_interaction = min_interactions
        self.sep = sep
        self.binary_classification = binary_classification
        self.rawData = self._load_data(path + data_file, sep)
        self.rawMeta = self._load_data(path + meta_file, sep)
        self.data = self.create_data(
            user_column,
            item_column,
            ratings_colum,
            min_interactions,
            ctx_categorical_columns,
            ctx_numerical_columns,
            columns_to_normalize,
            min_samples_per_user_test_set,
        )

    def _load_datasets(self, path, data_file, meta_file, sep):
        if data_file:
            self.rawData = self._load_data(path + data_file, sep)
        if meta_file:
            self.rawMeta = self._load_data(path + meta_file, sep)
        if data_file and meta_file:
            data = self._merge_data(self.rawData, self.rawMeta, self.item_column)[self.columns]
        
        
        



    def create_data(
        self,
        user_column: str,
        item_column: str,
        ratings_colum: str,
        min_interactions: int,
        ctx_categorical_columns: List[str],
        ctx_numerical_columns: List[str],
        columns_to_normalize: List[str],
        min_samples_per_user_test_set: int,
    ) -> DataFrame:
        

        data = self._merge_data(self.rawData, self.rawMeta, item_column)[self.columns]
        # TODO: We can move the clear rating at the end with before the train test split 
        data = self._clear_ratings(data, ratings_colum)
        # TODO: Create an attribute for that. 
        data = self._log_numerical(data, ctx_numerical_columns)
        data = self._normalize_columns(data, columns_to_normalize)
        data = self._initialize_iterative_cleaning(
            data, user_column, item_column, ratings_colum, min_interactions
        )
        data = self._update_elements_IDs(data)
        # TODO: Do we actually neeed it?
        data = data[self.columns]
        data = self.binarize_nominal_features(data, ctx_categorical_columns)
        self.train_ratings, self.test_ratings = (
            self._initialize_leave_one_out_train_test_split(
                data, min_samples_per_user_test_set
            )
        )
        # TODO: We could sort the data based on unser

        if self.binary_classification:
            for user in self.test_ratings[user_column]:
                pass

        return data

    def _load_data(self, data_path, sep) -> DataFrame:
        """
        Given a path, loads datasets
        """
        return pd.read_csv(data_path, sep=sep)

    def _merge_data(self, data, metadata, key_column) -> DataFrame:
        """
        Function that given 2 datasets it merges them
        """
        dataframe = data.merge(metadata, on=key_column)
        return dataframe

    def _clean_columns(self, *args: str | List[str] | Tuple[str]) -> List[str]:
        """
        Cleans and consolidates column names from various input formats into a single list of strings.

        This method processes input arguments that can either be individual string names of columns or collections
        of strings (e.g., lists or tuples) containing multiple column names. It ensures that all column names are
        collected into a flat list of strings, regardless of how they were passed to the method.

        Parameters:
        - args: Variable number of arguments, each can be a string representing a single column name or an iterable
        (e.g., list or tuple) of strings representing multiple column names.

        Returns:
        - List[str]: A list containing all column names as strings, with individual string arguments and elements
        from iterable arguments combined into a single list.
        """

        clean_columns = []
        for i in args:
            if isinstance(i, str):
                clean_columns.append(i)
                continue
            clean_columns.extend(i)
        return clean_columns

    def _clear_ratings(self, df: DataFrame, rating_column: str) -> DataFrame:
        df[rating_column] = pd.to_numeric(df[rating_column], errors="coerce")
        df = df.dropna(subset=[rating_column])
        return df

    def _log_numerical(self, df: DataFrame, numerical_features: List[str]) -> DataFrame:
        df_tmp = df.copy()
        for feature in numerical_features:
            df_tmp[feature] = df_tmp[feature].apply(
                lambda x: np.log10(x) if x != 0 else np.log10(x + 1)
            )
        return df_tmp

    def _normalize_columns(
        self, data: DataFrame, columns_to_normalize: List[str]
    ) -> DataFrame:
        """
        Function that normalizes a list of columns. It scales the data between 0 and 1 by applying sklearn MinMaxScaler

        Parameters:
            - data (DataFrame): Data frame which contains the columns to be normalized
            - columns_to_normalize (List[str]): A list containing the names of the columns to normalize

        Returns:
            - DataFrame: The dataframe with the normalized columns.
        """
        # Initializing the MinMaxScaler
        scaler = MinMaxScaler()
        # Applying the scaler to the selected columns
        data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

        return data

    def binarize_nominal_features(
        self, df: DataFrame, contextual_features: List[str]
    ) -> DataFrame:
        """
        Given a List of Contextual Features, in binarizes the features using one-hot enconding

        Parameters:
        - data (DataFrame): Dataset containing interactions.
        - contextual_features List[(str)]: The column name to filter by (either user or item).

        Returns:
        - DataFrame: The dataframe binirized.

        """
        df_bin = pd.get_dummies(df, columns=contextual_features, dtype=int)
        return df_bin

    def get_element_idx_given_min_interactions(
        self,
        data: DataFrame,
        filter_column: str,
        ratings_column: str,
        min_interactions: str,
    ):
        """
        Identify the indices of elements (users or items) with interactions less than or equal to a minimum threshold.

        Parameters:
        - data (DataFrame): Dataset containing interactions.
        - filter_column (str): The column name to filter by (either user or item).
        - interaction_column (str): The column that contains the interaction counts.
        - min_interactions (int): The threshold below which elements are considered to have few interactions.

        Returns:
        - numpy.ndarray: A boolean array where True indicates an element with few interactions.

        """
        data_groupedby_item_interaction = data.groupby(filter_column)[
            [ratings_column]
        ].count()
        iDs_low_interaction = data_groupedby_item_interaction[
            data_groupedby_item_interaction[ratings_column] <= min_interactions
        ].index

        idx_few_interactions = np.isin(data[filter_column], iDs_low_interaction)

        return idx_few_interactions

    def _initialize_iterative_cleaning(
        self,
        data: DataFrame,
        user_column: str,
        item_column: str,
        interaction_column: str,
        min_interactions: int,
    ) -> DataFrame:
        """
        Iteratively clean data by removing users and items with interactions below a certain threshold.

        Parameters:
        - data (DataFrame): The dataset to be cleaned.
        - user_column (str): The column name representing users.
        - item_column (str): The column name representing items.
        - interaction_column (str): The column name representing interactions.
        - min_interactions (int): Minimum number of interactions to not be filtered out.

        Returns:
        - DataFrame: The cleaned dataset.
        """

        is_stable = False
        iteration_count = 0
        data_iter = data.copy()
        while not is_stable:
            item_indx = self.get_element_idx_given_min_interactions(
                data_iter, item_column, interaction_column, min_interactions
            )
            data_clean_items = data_iter[~item_indx]
            user_indx = self.get_element_idx_given_min_interactions(
                data_clean_items, user_column, interaction_column, min_interactions
            )
            data_clean_items_users = data_clean_items[~user_indx]
            data_iter = data_clean_items_users.copy()
            min_item_interactions = (
                data_clean_items_users.groupby(item_column)[interaction_column]
                .count()
                .min()
            )
            min_user_interactions = (
                data_clean_items_users.groupby(user_column)[interaction_column]
                .count()
                .min()
            )

            if (
                min_item_interactions > min_interactions
                and min_user_interactions > min_interactions
            ):
                is_stable = True
            if iteration_count == 10:
                print(f"Iteration {iteration_count}")
            iteration_count += 1

        return data_clean_items_users

    def _initialize_train_test(self, data: DataFrame, min_samples_test_set: int):
        """
        Funtion that splits the dataser into train/test.
        Following the strategy of Leave one out - test set 1 interaction per user


        TODO: Update, we have the prefilter.
        """
        frequency_interaction = data.groupby(self.user_column)[
            [self.item_column, self.ratings_column]
        ].count()
        users_one_interaction = frequency_interaction[
            frequency_interaction[self.ratings_column] == 1
        ].index
        list_of_users = data[self.user_column].unique()
        users_more_one_interaction = list(
            set(list_of_users) - set(users_one_interaction)
        )
        idx = []
        for i in users_more_one_interaction:
            element = data[data["user"] == i].sample(n=min_samples_test_set)
            idx.append(element.index[0])

        test_idx = np.isin(data.index, np.array(idx))

        train_ratings, test_ratings = (
            data[~test_idx],
            data[test_idx],
        )
        return train_ratings, test_ratings

    def _initialize_leave_one_out_train_test_split(
        self, data: DataFrame, min_samples_test_set: int
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Function that splits the dataser into train/test.
        Following the strategy of Leave X out for the test set.
        If min_samples_test_set ==  1:
            The strategy follows a leave 1 out.
        If min_samples_test_set > 1:
            The number of interactions is checked, if the user has more, min_samples_test_set
            are alocated in the test set.

        Parameters:
            - data ((DataFrame): Raw or processed data to be treated
            - min_samples_test_set (int): The number of interactions to leave in the test set x user

        Return:
            - train_ratings (DataFrame): Training Dataset
            - test_ratings (DataFrame): Test Dataset
        """
        frequency_interaction = data.groupby(self.user_column)[self.item_column].count()

        list_of_users = data[self.user_column].unique()

        test_idx = []
        for user in list_of_users:
            num_interactions = frequency_interaction[user]
            if num_interactions > (
                min_samples_test_set + 1
            ):  # should have at least one for the train test
                element = data[data[self.user_column] == user].sample(
                    n=min_samples_test_set
                )
            elif num_interactions > 2:
                element = data[data[self.user_column] == user].sample(n=1)

            test_idx.extend(element.index)

        test_mask = np.isin(data.index, np.array(test_idx))

        train_ratings, test_ratings = (
            data[~test_mask],
            data[test_mask],
        )
        return train_ratings, test_ratings

    def _initialize_standard_train_test_split(
        self, data: DataFrame, test_size: float, random_state: int
    ) -> Tuple[DataFrame, DataFrame]:
        """ "
         Performs a train test split on the data set using sklearn train test split.

        Parameters:
             - data ((DataFrame): Raw or processed data to be treated
             - min_samples_test_set (int): The number of interactions to leave in the test set x user

         Return:
             - train_ratings (DataFrame): Training Dataset
             - test_ratings (DataFrame): Test Dataset
        """
        X_train, X_test = train_test_split(
            data, test_size=test_size, random_state=random_state
        )
        return X_train, X_test

    def _map_elementIDs(self, data: DataFrame, column_to_map: str) -> Dict:
        """
        Function that takes the non continous sequence of unique IDs from a column, i.e., UserId, ItemID and maps it to a continuous sequence.
        It stores the index map into a dictionary.

         Parameters:
        - data (DataFrame): The dataset to be extract unique Id.
        - column_to_map (str): The column name which contains the non continous sequence of unique IDs.

        Returns:
        - element_id_to_index (Dict): A dictionary containing the element_id: new index
        """
        unique_element_ids = sorted(data[column_to_map].unique())
        element_id_to_index = {
            element_id: index for index, element_id in enumerate(unique_element_ids)
        }
        return element_id_to_index

    def _update_elements_IDs(self, data: DataFrame) -> DataFrame:
        """
        Ensures user and item IDs in the DataFrame are continuous sequences. If not,
        remaps the IDs to be continuous.

        Parameters:
        - data (DataFrame): The input data with user and item columns.

        Returns:
        - DataFrame: The modified DataFrame with continuous user and item ID sequences.
        """
        unique_user_id = data[self.user_column].unique()
        unique_item_id = data[self.item_column].unique()

        # Check if the sequences are continuous
        is_user_id_sequence_continuous = (
            max(unique_user_id) - min(unique_user_id) + 1
        ) == len(unique_user_id)
        is_item_id_sequence_continuous = (
            max(unique_item_id) - min(unique_item_id) + 1
        ) == len(unique_item_id)

        if is_user_id_sequence_continuous and is_item_id_sequence_continuous:
            return data

        # Mapping elements
        user_id_to_index = self._map_elementIDs(data, self.user_column)
        item_id_to_index = self._map_elementIDs(data, self.item_column)

        # Apply mappings
        data[self.user_column] = data[self.user_column].map(user_id_to_index)
        data[self.item_column] = data[self.item_column].map(item_id_to_index)

        return data

    def save_data(self, folder_name: str) -> None:
        """
        Saves the Class atributes train-ratings, under the path data/processed/folder_name/
        the files have the following format:
            - folder_name.train.ratings
            - folder_name.test.ratings

        Parameters:
            - folde_name (str): The folder name under the processed data will be saved
        """
        current_file_path = os.path.abspath(__file__)
        data_folder_path = os.path.dirname(current_file_path)
        processed_data_path = os.path.join(data_folder_path, "processed", folder_name)
        folder_path = os.path.join(processed_data_path, folder_name)

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
        self.create_data_info(processed_data_path)
        print("Saved in: ", folder_path)

    def create_data_info(self, processed_data_path):
        num_users_raw = len(self.rawData[self.user_column].unique())
        num_items_raw = len(self.rawData[self.item_column].unique())
        num_interactions_raw = self.rawData.shape[0]

        num_users_processed = len(self.data[self.user_column].unique())
        num_items_processed = len(self.data[self.item_column].unique())
        num_interactions_processed = self.data.shape[0]

        ratio_reduction_users = (
            (num_users_raw - num_users_processed) / num_users_raw * 100
        )
        ratio_reduction_items = (
            (num_items_raw - num_items_processed) / num_items_raw * 100
        )
        ratio_reductions_interactions = (
            (num_interactions_raw - num_interactions_processed)
            / num_interactions_raw
            * 100
        )

        content = textwrap.dedent(
            f"""\
        Processed dataset: Frappe \n
        Items and Users with less than {self.min_interaction} interactions have been removed.
        Number of Users: {num_users_processed}.
          - Reduction: {ratio_reduction_users:.2f}%.
        Number of Items: {num_items_processed}.
          - Reduction: {ratio_reduction_items:.2f}%.
        Total number of interactions: {num_interactions_processed}
          - Reduction: {ratio_reductions_interactions:.2f}%. 
        Columns used: {', '.join(self.columns)}."""
        )

        try:
            with open(processed_data_path + "/ReadMe.txt", "w") as file:
                file.write(content)
                file.write("\n\nColumns details:\n")
                for i, col in enumerate(self.data.columns):
                    file.write(f"{i} - {col}\n")
        except Exception as e:
            raise RuntimeError(f"An error occurred: {e}")

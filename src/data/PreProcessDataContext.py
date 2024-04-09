import errno
import os
import pickle
import textwrap
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class PreProcessDataNCFContextual:
    """
    Preprocess script.
    Given a data set with categorical and numercial features the script performs filtering,
    data transformation, normalization, and one-hot encoding on the dataset.

    Parameters:
    - columns_to_transform (dict): A dictionary containg the transofrmation and the columns to transform
    {
        'log': ['cnt'],
        'cyclical':['weeknumber', 'friends']
    }

    """

    def __init__(
        self,
        path: str,
        data_file: str = None,
        meta_file: str = None,
        key_column: str = None,
        user_column: str = None,
        item_column: str = None,
        ratings_column: str = None,
        ctx_categorical_columns: List[str] = None,
        ctx_numerical_columns: List[str] = None,
        columns_to_transform: Dict[str, str | List[str]] = None,
        columns_to_normalize: List[str] = None,
        folder_name: str = None,
        min_interactions: int = 5,
        min_samples_per_user_test_set: int = 1,
        is_binary_classification: bool = True,
        train_test_split: str = "loo",
        sep: str = "\t",
        test_size: int = 0.2,
    ) -> None:

        # Check of required parameters
        self._check_required_paremeters(
            path=path,
            data_file=data_file,
            user_column=user_column,
            item_column=item_column,
            ratings_column=ratings_column,
        )
        self.columns_to_transform = columns_to_transform or {}
        self.encodings = {
            "log": self._logarithmic_encoding,
            "cyclical": self._cyclical_encoding_inplace,
        }
        self._check_transformation_functions()

        self.path = path
        self.ratings_column = ratings_column
        self.user_column = user_column
        self.item_column = item_column
        self.min_interaction = min_interactions
        self.sep = sep
        self.is_binary_classification = is_binary_classification
        self.train_test_split = train_test_split
        self.test_size = test_size
        self.key_column = key_column

        self.columns = self._clean_columns(
            user_column,
            item_column,
            ratings_column,
            ctx_categorical_columns,
            ctx_numerical_columns,
        )
        self._load_datasets(
            path=path,
            meta_file=meta_file,
            data_file=data_file,
            sep=self.sep,
        )
        self.data = self.create_data(
            user_column=user_column,
            item_column=item_column,
            ratings_colum=ratings_column,
            min_interactions=min_interactions,
            ctx_categorical_columns=ctx_categorical_columns,
            ctx_numerical_columns=ctx_numerical_columns,
            columns_to_transform=columns_to_transform,
            columns_to_normalize=columns_to_normalize,
            min_samples_per_user_test_set=min_samples_per_user_test_set,
        )

    def create_data(
        self,
        user_column: str,
        item_column: str,
        ratings_colum: str,
        min_interactions: int,
        ctx_categorical_columns: List[str] | None,
        ctx_numerical_columns: List[str] | None,
        columns_to_transform: Dict[str, List[str]] | None,
        columns_to_normalize: List[str] | None,  # This should be all Numerical Columns
        min_samples_per_user_test_set: int,
    ) -> DataFrame:

        # Load Data
        data = self._merge_data_frames(item_column)

        # Numerical Transformations
        if columns_to_transform:
            for encoding in columns_to_transform.keys():
                data = self.encodings[encoding](data, columns_to_transform[encoding])
                print(
                    f"{encoding} transformation performed on {columns_to_transform[encoding]} "
                )
            # data = self._logarithmic_encoding(data, ctx_numerical_columns)
        if columns_to_normalize:
            data = self._normalize_columns(data, columns_to_normalize)
        elif ctx_numerical_columns:
            warnings.warn(
                "\nNo Columns specified to normalize: Normalizing all Numerical Columns"
            )
            data = self._normalize_columns(data, ctx_numerical_columns)
        else:
            warnings.warn("No Numerical columns have been paseed")

        # Data Cleaning
        data = self._initialize_iterative_cleaning(
            data, user_column, item_column, ratings_colum, min_interactions
        )
        print(f"K-core cleaning performed with k: {min_interactions}")
        # data = self._update_elements_IDs(data)
        data = self._update_elements_IDs_factorized(data)

        data = self._clear_ratings(data, ratings_colum)

        self.positive_samples = self._create_positive_sampling(data)

        # One-hot encoding
        if ctx_categorical_columns:
            data = self._initialize_one_hot_encoding(data, ctx_categorical_columns)

        # Binary clasificaton
        if self.is_binary_classification:
            data[self.ratings_column] = data[self.ratings_column].where(
                data[self.ratings_column] <= 0, 1
            )

        # Train Test Split
        if self.train_test_split == "loo":
            self.train_ratings, self.test_ratings = (
                self._initialize_leave_one_out_train_test_split(
                    data=data, min_samples_test_set=min_samples_per_user_test_set
                )
            )

        elif self.train_test_split == "standard":
            self.train_ratings, self.test_ratings = (
                self._initialize_standard_train_test_split(
                    data=data, test_size=self.test_size
                )
            )

        return data

    def _create_positive_sampling(self, data: DataFrame) -> dict:
        """
        Given a dataset it computes a dictionary with the interacted Items per user

        Parameters:
            - data (DataFrame): The data frame containg at least 2 columns User and Item
        Returns:
            - interacted_items (dict): Dictionary with users as keys and interated items per user as values
        """
        interacted_items = (
            data.groupby([self.user_column])[self.item_column].apply(set).to_dict()
        )
        return interacted_items

    def _merge_data_frames(self, item_column):
        # Check if both DataFrames are loaded and not empty
        if (
            self.rawData is not None
            and not self.rawData.empty
            and self.rawMeta is not None
            and not self.rawMeta.empty
        ):
            # Check if a key column has been specified
            if not self.key_column:
                warnings.warn(
                    f"No Key Column specified. Using {item_column} column as Key column.",
                )
                try:
                    data = self._merge_data(
                        self.rawData, self.rawMeta, key_column=item_column
                    )[self.columns].copy()
                    return data
                except KeyError as e:
                    raise KeyError("Item column cannot be used as key column") from e
            else:
                data = self._merge_data(
                    self.rawData, self.rawMeta, key_column=self.key_column
                )[self.columns].copy()
        else:
            if self.rawData is not None and not self.rawData.empty:
                data = self.rawData[self.columns].copy()
            else:
                raise AttributeError("No DataFrame loaded")
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

    def _clean_columns(self, *args: str | List[str] | Tuple[str] | None) -> List[str]:
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
            if not i:
                continue
            if isinstance(i, str):
                clean_columns.append(i)
                continue
            clean_columns.extend(i)
        return clean_columns

    def _clear_ratings(self, df: DataFrame, rating_column: str) -> DataFrame:
        df[rating_column] = pd.to_numeric(df[rating_column], errors="coerce")
        df = df.dropna(subset=[rating_column])
        return df

    def _cyclical_encoding_inplace(self, df: DataFrame, numerical_features: List[str]):
        for feature in numerical_features:
            max_value = df[feature].max()
            # Use .loc to ensure modifications are done on the original DataFrame
            df.loc[:, f"sin_{feature}"] = np.sin((2 * np.pi * df[feature]) / max_value)
            df.loc[:, f"cos_{feature}"] = np.cos((2 * np.pi * df[feature]) / max_value)
        return df

    def _logarithmic_encoding(
        self, df: DataFrame, numerical_features: List[str]
    ) -> DataFrame:
        """
        Applies logarithmic base 10 encoding to numerical features in a DataFrame.

        Parameters:
        - df (DataFrame): The pandas DataFrame containing the data to be transformed.
        - numerical_features (List[str]): A list of column names in `df` that correspond to the numerical
        features to be log-transformed.

        Returns:
        - DataFrame: The modified DataFrame with log-transformed numerical features.
        """
        for feature in numerical_features:
            df[feature] = np.log10(df[feature] + 1)
        return df

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
        data[columns_to_normalize] = data[columns_to_normalize].astype(float)
        # Applying the scaler to the selected columns
        data.loc[:, columns_to_normalize] = scaler.fit_transform(
            data[columns_to_normalize].values
        )

        return data

    def _initialize_one_hot_encoding(
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
            print(f"Iteration {iteration_count}")
            if iteration_count == 10:
                print(f"Iteration {iteration_count}")
            iteration_count += 1

        return data_clean_items_users

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

    def _update_elements_IDs_factorized(self, data: DataFrame) -> DataFrame:
        """
        Ensures user and item IDs in the DataFrame are continuous sequences. If not,
        remaps the IDs to be continuous using pandas' factorize() method.

        Parameters:
        - data (DataFrame): The input data with user and item columns.

        Returns:
        - DataFrame: The modified DataFrame with continuous user and item ID sequences.
        """
        # Factorize user and item IDs
        data[self.user_column], _ = pd.factorize(data[self.user_column])
        data[self.item_column], _ = pd.factorize(data[self.item_column])

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
        processed_data_path, folder_path = self.create_save_directory(folder_name)

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
        try:
            with open(folder_path + ".positive_samples.pkl", "wb") as f:
                pickle.dump(self.positive_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise RuntimeError(f"An error occurred: {e}")

        self.create_data_info(processed_data_path)
        print("Saved in: ", folder_path)

    def create_save_directory(self, folder_name):
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
        return processed_data_path, folder_path

    def create_data_info(self, processed_data_path):
        # TODO: We can calculate this right after loading data and then erase raw data
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

    def _check_transformation_functions(self):
        for enc in self.columns_to_transform.keys():
            # Check if the item is a string and convert it to a list if it is
            if isinstance(self.columns_to_transform[enc], str):
                self.columns_to_transform[enc] = [self.columns_to_transform[enc]]

            # Check if the encoding specified is in the current encodings
            if enc not in self.encodings.keys():
                content = dict()
                for i in self.encodings.keys():
                    content[i] = self.encodings[i].__doc__
                raise ValueError(
                    f"The function you are trying to perform is not yet developed. Possible encodings: \n{content}"
                )

    def _check_required_paremeters(self, **kwargs):
        """
        Given a set of parameters, it it throws an error if one of them is missing.

        Parameters:
         - kwargs: Keyword arguments representing parameters. Each parameter is a string.
        """
        # Check if any of the required parameters is None
        missing_params = [param for param, value in kwargs.items() if value is None]
        if missing_params:
            raise ValueError(
                f"Missing required parameters: {', '.join(missing_params)}"
            )

    def _load_datasets(self, path, data_file, meta_file, sep):

        if data_file:
            self.rawData = self._load_data(path + data_file, sep=sep)
        else:
            self.rawData = None
        if meta_file:
            self.rawMeta = self._load_data(path + meta_file, sep)
        else:
            self.rawMeta = None

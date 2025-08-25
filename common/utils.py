import os
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from enum import Enum
import torch
import random


class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    VALIDATION = 2


def load_dataset(
    name: str, dataset_type: DatasetType = DatasetType.TRAIN, index: bool = True
) -> pd.DataFrame:
    """
    Loads a dataset from the specified data repository.

    This function constructs the path to a dataset based on the provided `name` and
    `dataset_type`.
    It retrieves the base data repository path from the environment variable `DATA_REPOSITORY`.
    If the environment variable is not set, an `EnvironmentError` is raised.

    Args:
        name (str): The name of the dataset to load (e.g., "dataset_name").
        dataset_type (DatasetType, optional): The type of dataset to load. Can be one of
            `DatasetType.TRAIN`, `DatasetType.TEST`, or `DatasetType.VALIDATION`. Defaults to
            `DatasetType.TRAIN`.
        index (bool, optional): Whether to set the first column of the CSV file as the index.
        Defaults to `True`.

    Raises:
        EnvironmentError: If the environment variable `DATA_REPOSITORY` is not set.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded dataset.

    Example:
        df = load_dataset("my_dataset", dataset_type=DatasetType.TEST, index=False)
    """

    dataset_path = os.getenv("DATA_REPOSITORY")

    if dataset_path is None:
        raise EnvironmentError("Required environment variable 'DATA_REPOSITORY' is not set.")

    dataset_path = Path(dataset_path, name)

    if dataset_type == DatasetType.TRAIN:
        dataset_path = Path(dataset_path, "train.csv")
    elif dataset_type == DatasetType.TEST:
        dataset_path = Path(dataset_path, "test.csv")
    elif dataset_type == DatasetType.VALIDATION:
        dataset_path = Path(dataset_path, "validation.csv")

    if index:
        return pd.read_csv(dataset_path)
    else:
        return pd.read_csv(dataset_path, index_col=[0])


def optimize_memory(dataframe, deep=False):
    """
    Optimizes memory usage of a pandas DataFrame by converting columns to more efficient data
    types.

    This function iterates over the columns of the input DataFrame (`props`) and attempts to
    downcast
    numerical columns to the smallest possible integer or float data types without losing
    information.
    If a column contains missing values (NaNs), it will be filled before attempting to convert
    the data type.

    Args:
        dataframe (pandas.DataFrame): The DataFrame whose memory usage needs to be optimized.
        deep (bool, optional): Whether or not to perform a deep memory usage calculation.
        Defaults to False.

    Returns:
        pandas.DataFrame: The input DataFrame with optimized memory usage.
        list: A list of column names where missing values were filled.

    Raises:
        None: This function does not raise any exceptions.

    Example:
        optimized_df, na_columns = optimize_memory(df)
    """
    start_mem_usg = dataframe.memory_usage(deep=deep).sum() / 1024**2

    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")

    na_list = []  # Keeps track of columns that have missing values filled in.

    for col in dataframe.columns:
        if dataframe[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", dataframe[col].dtype)

            # make variables for Int, max and min
            is_int = False
            mx = dataframe[col].max()
            mn = dataframe[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(dataframe[col]).all():
                na_list.append(col)
                dataframe[col].fillna(mn - 1, inplace=True)

            # test if column can be converted to an integer
            asint = dataframe[col].fillna(0).astype(np.int64)
            result = dataframe[col] - asint
            result = result.sum()
            if -0.01 < result < 0.01:
                is_int = True

            # Make Integer/unsigned Integer datatypes
            if is_int:
                if mn >= 0:
                    if mx < 255:
                        dataframe[col] = dataframe[col].astype(np.uint8)
                    elif mx < 65535:
                        dataframe[col] = dataframe[col].astype(np.uint16)
                    elif mx < 4294967295:
                        dataframe[col] = dataframe[col].astype(np.uint32)
                    else:
                        dataframe[col] = dataframe[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        dataframe[col] = dataframe[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        dataframe[col] = dataframe[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        dataframe[col] = dataframe[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        dataframe[col] = dataframe[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                dataframe[col] = dataframe[col].astype(np.float32)

            # Print new column type
            print("dtype after: ", dataframe[col].dtype)
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = dataframe.memory_usage().sum() / 1024**2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return dataframe, na_list


def get_params(model: XGBRegressor | XGBClassifier):
    """
    Retrieves the hyperparameters of a given XGBoost model (either XGBRegressor or XGBClassifier).

    This function calls the `get_params()` method on the provided XGBoost model and returns the
    parameters in a pandas DataFrame for easy inspection and analysis. The DataFrame contains two
    columns: 'Parameter' (the name of the hyperparameter) and 'Value' (the corresponding value).

    Args:
        model (XGBRegressor | XGBClassifier): The XGBoost model (either a regressor or classifier)
            from which to retrieve the hyperparameters.

    Returns:
        pandas.DataFrame: A DataFrame containing the hyperparameters and their corresponding
        values.

    Example:
        model = XGBRegressor()
        params_df = get_params(model)
        print(params_df)
    """
    return pd.DataFrame(list(model.get_params().items()), columns=["Parameter", "Value"])

import os
import tomllib
from pathlib import Path

import pandas as pd
import plotly.express as px


def load_dataset(dataset_name) -> pd.DataFrame:
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
        dataset_path = Path(data["project"]["data_repository_linux"], dataset_name, "train.csv")

        if os.name == "nt":
            dataset_path = Path(data["project"]["data_repository_windows"], dataset_name,
                                "train.csv")

        return pd.read_csv(dataset_path)
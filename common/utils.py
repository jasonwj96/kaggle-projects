import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px

def plot_missing_values(dataframe: pd.DataFrame, missing_percent: float = 0.05):
    missing_values = dataframe.isnull().sum()
    missing_values = dataframe.isnull().sum() / (dataframe.isnull().sum() + dataframe.count())
    missing_values = missing_values[missing_values > missing_percent]

    fig = px.bar(data_frame=missing_values, 
                title='Percent Missing Values by Column')
    fig.update_layout(template='plotly_dark')

    fig.show()
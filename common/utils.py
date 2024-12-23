import pandas as pd
import plotly.express as px

def plot_missing_values(dataframe: pd.DataFrame, missing_percent: float = 0.05):
    missing_values = dataframe.isnull().sum()
    missing_values = dataframe.isnull().sum() / (dataframe.isnull().sum() + dataframe.count())
    missing_values = missing_values[missing_values > missing_percent]

    fig = px.bar(data_frame=missing_values, 
                title='Percent Missing Values by Column')
    fig.update_layout(template='plotly_dark')

    fig.show()
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pyreadr import read_r
import numpy as np
import plotly.express as px
import janitor as jn


def clean_data(df):
    """
    This functions takes a dataframe and first ensures that all states are valid US States. It removes observations corresponding to non-valid states.

    Then Replace the 0's in the answers as NaN's. These correspnd to unanswered questions.


    Parameters:
    df (pd.DataFrame): The dataframe to clean.

    Returns:
    pd.DataFrame: The cleaned dataframe.

    """
    # Deine Valid States
    valid_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN',
                    'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

    df = df[df['STATE'].isin(valid_states)]

    #  Replace 0's in the answers as NaN's. These correspnd to unanswered questions.

    df = df.replace(0, np.nan)

    return df

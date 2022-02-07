import numpy as np
import pandas as pd


def process(
    data,
    missing_value=[],
    features_to_drop=[],
    categorical_features=[],
):
    result = []

    df = data

    # drop nan values
    df = df.replace(missing_value, np.nan)
    df = df.dropna(axis=0)

    # drop useless features
    df = df.drop(columns=features_to_drop)

    # create one-hot encoding of categorical features
    df = pd.get_dummies(df, columns=categorical_features, prefix_sep="=")

    df = df.reset_index(drop=True)

    return df

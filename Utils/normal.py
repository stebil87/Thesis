import numpy as np
import pandas as pd

def z_score(df, target='y'):
    df_normalized = df.copy()
    target_column = df_normalized[target]
    features = df_normalized.drop(columns=[target], errors='ignore')
    
    row_means = features.mean(axis=1)
    row_stds = features.std(axis=1)
    
    features_normalized = features.sub(row_means, axis=0).div(row_stds, axis=0)
    
    df_normalized = pd.concat([features_normalized, target_column], axis=1)
    return df_normalized

def normalize(dataframes):
    normalized_dataframes = {}
    for name, df in dataframes.items():
        normalized_df = z_score(df, target='y')
        normalized_dataframes[name] = normalized_df
    return normalized_dataframes